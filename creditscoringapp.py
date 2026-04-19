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
This app predicts the likelihood of a customer defaulting on a loan using Logistic Regression and XGBoost models.
Predictions are loaded from Cloudflare R2 or uploaded CSV files.
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
# Data Cleaning
# ---------------------------
def clean_numeric_columns(df):
    return df.applymap(
        lambda x: float(str(x)
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")) if isinstance(x, str) else x
    )

# ---------------------------
# Batch placeholder
# ---------------------------
batch = None

# ---------------------------
# Load Data from Cloudflare R2
# ---------------------------
data_df = None

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

    data_df = clean_numeric_columns(data_df)
    data_df.fillna(data_df.median(numeric_only=True), inplace=True)

    # Feature Engineering
    data_df["TotalPastDue"] = (
        data_df["NumberOfTime30-59DaysPastDueNotWorse"] +
        data_df["NumberOfTime60-89DaysPastDueNotWorse"] +
        data_df["NumberOfTimes90DaysLate"]
    )

    data_df["DebtPerIncome"] = data_df["DebtRatio"] * data_df["MonthlyIncome"]

    data_df = data_df[FEATURE_COLUMNS]

    st.success("Dataset loaded from Cloudflare R2")

except Exception as e:
    st.warning(f"Could not load dataset: {e}")
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
st.subheader("Predictions on Cloudflare Dataset")

features_df = data_df.copy()

scaled_data = scaler.transform(features_df)

data_df["LogReg_Prob"] = logreg_model.predict_proba(scaled_data)[:, 1]
data_df["XGB_Prob"] = xgb_model.predict_proba(features_df)[:, 1]

st.dataframe(data_df)
st.download_button("Download Predictions", data_df.to_csv(index=False), "predictions.csv")

# ---------------------------
# SHAP Business Interpretation
# ---------------------------
st.subheader("Business Interpretation (XGBoost)")

try:
    # Sample selection
    if batch is not None and all(col in batch.columns for col in FEATURE_COLUMNS):
        sample_row = batch[FEATURE_COLUMNS].iloc[[0]]
        background = batch[FEATURE_COLUMNS].sample(min(50, len(batch)))
    else:
        sample_row = data_df[FEATURE_COLUMNS].median().to_frame().T
        background = data_df[FEATURE_COLUMNS].sample(min(50, len(data_df)))

    # ---------------------------
    # OPTIMAL SHAP (TreeExplainer)
    # ---------------------------
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample_row)

    # Convert safely to array
    shap_values = np.array(shap_values)

    # ---------------------------
    # Waterfall Plot
    # ---------------------------
    shap.initjs()
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
    # Feature Importance Table
    # ---------------------------
    feature_impact = pd.DataFrame(
        list(zip(FEATURE_COLUMNS, shap_values[0])),
        columns=["Feature", "SHAP_Value"]
    ).sort_values(by="SHAP_Value", key=abs, ascending=False)

    st.markdown("**Top drivers of default risk (XGBoost):**")

    for _, row in feature_impact.head(3).iterrows():
        if row["SHAP_Value"] > 0:
            meaning = "INCREASES default risk"
        else:
            meaning = "DECREASES default risk"

        st.write(
            f"- {row['Feature']} {meaning} (impact: {row['SHAP_Value']:.4f})"
        )

except Exception as e:
    st.warning(f"Business Interpretation failed: {e}")

# ---------------------------
# Batch Upload Predictions
# ---------------------------
st.subheader("Upload CSV for Batch Predictions")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    batch = pd.read_csv(file)
    batch = clean_numeric_columns(batch)
    batch.fillna(batch.median(numeric_only=True), inplace=True)

    batch["TotalPastDue"] = (
        batch["NumberOfTime30-59DaysPastDueNotWorse"] +
        batch["NumberOfTime60-89DaysPastDueNotWorse"] +
        batch["NumberOfTimes90DaysLate"]
    )

    batch["DebtPerIncome"] = batch["DebtRatio"] * batch["MonthlyIncome"]

    batch_features = batch[FEATURE_COLUMNS]

    batch_scaled = scaler.transform(batch_features)

    batch["LogReg_Prob"] = logreg_model.predict_proba(batch_scaled)[:, 1]
    batch["XGB_Prob"] = xgb_model.predict_proba(batch_features)[:, 1]

    st.dataframe(batch)
    st.download_button(
        "Download Batch Predictions",
        batch.to_csv(index=False),
        "batch_predictions.csv"
    )