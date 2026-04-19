import streamlit as st
import pandas as pd
import numpy as np
import joblib
import boto3
from io import BytesIO
import shap
import matplotlib.pyplot as plt

# ---------------------------
# Streamlit Setup
# ---------------------------
st.set_page_config(page_title="Credit Scoring System", layout="wide")
st.title("Credit Scoring & Loan Decision System")

st.markdown("""
This app predicts loan default risk using Logistic Regression and XGBoost.
Data is loaded from Cloudflare R2 or uploaded as CSV.
""")

# ---------------------------
# Feature Columns
# ---------------------------
FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "Age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "TotalPastDue",
    "DebtPerIncome"
]

# ---------------------------
# CLEANING FUNCTIONS (CRITICAL FIX)
# ---------------------------
def clean_numeric(df):
    """Remove corrupted numeric strings like '[5E-1]' and convert safely"""
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str)
            .str.replace(r"[\[\]'\"]", "", regex=True),
            errors="coerce"
        )
    return df


def feature_engineering(df):
    df = df.copy()

    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"] +
        df["NumberOfTimes90DaysLate"]
    )

    df["DebtPerIncome"] = df["DebtRatio"] * df["MonthlyIncome"]

    return df

# ---------------------------
# Load R2 Data
# ---------------------------
try:
    s3 = boto3.client(
        "s3",
        endpoint_url=st.secrets["R2_ENDPOINT_URL"],
        aws_access_key_id=st.secrets["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["R2_SECRET_ACCESS_KEY"]
    )

    bucket = st.secrets["R2_BUCKET_NAME"]

    objects = s3.list_objects_v2(Bucket=bucket)
    file_name = next(obj["Key"] for obj in objects["Contents"] if obj["Key"].endswith(".csv"))

    obj = s3.get_object(Bucket=bucket, Key=file_name)
    data_df = pd.read_csv(BytesIO(obj["Body"].read()))

    data_df = clean_numeric(data_df)
    data_df = feature_engineering(data_df)

    data_df.fillna(data_df.median(numeric_only=True), inplace=True)

    data_df = data_df[FEATURE_COLUMNS]

    st.success("Dataset loaded successfully from Cloudflare R2")

except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

# ---------------------------
# Load Models
# ---------------------------
try:
    logreg_model = joblib.load("models/logreg_v2.pkl")
    xgb_model = joblib.load("models/xgb_best.pkl")
    scaler = joblib.load("models/scaler_v2.pkl")

    st.success("Models loaded successfully")

except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# ---------------------------
# Predictions (Cloud Data)
# ---------------------------
st.subheader("Cloud Dataset Predictions")

features_df = data_df.copy()

scaled = scaler.transform(features_df)

data_df["LogReg_Prob"] = logreg_model.predict_proba(scaled)[:, 1]
data_df["XGB_Prob"] = xgb_model.predict_proba(features_df)[:, 1]

st.dataframe(data_df)
st.download_button("Download Predictions", data_df.to_csv(index=False), "predictions.csv")

# ---------------------------
# Batch Upload
# ---------------------------
st.subheader("Upload CSV for Batch Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

batch = None

if file:
    batch = pd.read_csv(file)

    batch = clean_numeric(batch)
    batch = feature_engineering(batch)
    batch.fillna(batch.median(numeric_only=True), inplace=True)

    batch_features = batch[FEATURE_COLUMNS]

    batch_scaled = scaler.transform(batch_features)

    batch["LogReg_Prob"] = logreg_model.predict_proba(batch_scaled)[:, 1]
    batch["XGB_Prob"] = xgb_model.predict_proba(batch_features)[:, 1]

    st.dataframe(batch)
    st.download_button("Download Batch", batch.to_csv(index=False), "batch_predictions.csv")

# ---------------------------
# SHAP EXPLANATION (FIXED + OPTIMAL)
# ---------------------------
st.subheader("Business Interpretation (XGBoost)")

try:
    if batch is not None:
        sample_row = batch[FEATURE_COLUMNS].iloc[[0]]
        background = batch[FEATURE_COLUMNS].sample(min(50, len(batch)))
    else:
        sample_row = data_df[FEATURE_COLUMNS].median().to_frame().T
        background = data_df[FEATURE_COLUMNS].sample(min(50, len(data_df)))

    # FORCE CLEAN NUMERIC (IMPORTANT FIX FOR [5E-1] ISSUE)
    sample_row = clean_numeric(sample_row).astype(float)
    background = clean_numeric(background).astype(float)

    # SHAP (OPTIMAL FOR XGBOOST)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample_row)

    shap_values = np.array(shap_values)

    # ---------------------------
    # Waterfall Plot (Correct Streamlit rendering)
    # ---------------------------
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=sample_row.iloc[0],
            feature_names=FEATURE_COLUMNS
        ),
        show=False
    )

    st.pyplot(plt.gcf())
    plt.clf()

    # ---------------------------
    # Feature Impact Table
    # ---------------------------
    feature_impact = pd.DataFrame(
        list(zip(FEATURE_COLUMNS, shap_values[0])),
        columns=["Feature", "SHAP_Value"]
    ).sort_values(by="SHAP_Value", key=abs, ascending=False)

    st.markdown("### Top Drivers of Default Risk")

    for _, row in feature_impact.head(3).iterrows():
        if row["SHAP_Value"] > 0:
            meaning = "INCREASES default risk"
        else:
            meaning = "DECREASES default risk"

        st.write(f"- **{row['Feature']}** {meaning} (impact: {row['SHAP_Value']:.4f})")

except Exception as e:
    st.warning(f"Business Interpretation failed: {e}")