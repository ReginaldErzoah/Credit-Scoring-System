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
This app predicts loan default risk using Logistic Regression and XGBoost models.
Data is loaded from Cloudflare R2, and users can upload CSV files for batch scoring.
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
]

# ---------------------------
# CLEANING FUNCTION (for R2 + batch)
# ---------------------------
def clean_numeric_columns(df):
    return df.applymap(
        lambda x: float(
            str(x)
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
        )
        if isinstance(x, str) else x
    )

# ---------------------------
# STRICT SHAP CLEANING 
# ---------------------------
def force_numeric(df):
    return df.applymap(
        lambda x: pd.to_numeric(
            str(x).replace("[", "").replace("]", ""),
            errors="coerce"
        )
    )

# ---------------------------
# Load R2 Data
# ---------------------------
try:
    R2_ENDPOINT = st.secrets["R2_ENDPOINT_URL"]
    R2_ACCESS_KEY = st.secrets["R2_ACCESS_KEY_ID"]
    R2_SECRET_KEY = st.secrets["R2_SECRET_ACCESS_KEY"]
    R2_BUCKET = st.secrets["R2_BUCKET_NAME"]

    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY
    )

    objects = s3.list_objects_v2(Bucket=R2_BUCKET)
    file_name = next((obj['Key'] for obj in objects['Contents'] if obj['Key'].endswith('.csv')), None)

    obj = s3.get_object(Bucket=R2_BUCKET, Key=file_name)
    data_df = pd.read_csv(BytesIO(obj['Body'].read()))

    data_df = clean_numeric_columns(data_df)
    data_df.fillna(data_df.median(), inplace=True)

    data_df = data_df[FEATURE_COLUMNS]

    st.success("Dataset loaded successfully")

except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

# ---------------------------
# Load Models
# ---------------------------
try:
    logreg_model = joblib.load("models/logreg_v3.pkl")
    xgb_model = joblib.load("models/xgb_best.pkl")
    scaler = joblib.load("models/scaler_v3.pkl")

    st.success("Models loaded successfully")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------------------------
# Predictions
# ---------------------------
st.subheader("Predictions on Dataset")

features_df = data_df.copy()

scaled_data = scaler.transform(features_df)

data_df["LogReg_Prob"] = logreg_model.predict_proba(scaled_data)[:, 1]
data_df["XGB_Prob"] = xgb_model.predict_proba(features_df)[:, 1]

st.dataframe(data_df)
st.download_button(
    "Download Predictions",
    data_df.to_csv(index=False),
    "predictions.csv"
)

# ---------------------------
# BUSINESS INTERPRETATION (FIXED SHAP)
# ---------------------------
st.subheader("Business Interpretation (XGBoost)")

try:
    # STEP 1: Clean SHAP input separately (fixes [5E-1] issue)
    shap_data = force_numeric(data_df[FEATURE_COLUMNS].copy())

    shap_data = shap_data.dropna()

    # STEP 2: Safe sampling
    sample_row = shap_data.sample(1)
    background = shap_data.sample(min(50, len(shap_data)))

    # STEP 3: Proper SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample_row)

    # STEP 4: Build SHAP explanation object
    shap_exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=sample_row.iloc[0],
        feature_names=FEATURE_COLUMNS
    )

    # STEP 5: Plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_exp, show=False)
    st.pyplot(fig)

    # STEP 6: Feature importance
    feature_impact = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "SHAP_Value": shap_values[0]
    }).sort_values(by="SHAP_Value", key=abs, ascending=False)

    st.markdown("**Top 3 features influencing prediction:**")

    for _, row in feature_impact.head(3).iterrows():
        direction = "increases" if row["SHAP_Value"] > 0 else "decreases"
        st.write(
            f"- {row['Feature']} {direction} risk (impact: {row['SHAP_Value']:.3f})"
        )

except Exception as e:
    st.warning(f"Business Interpretation failed: {e}")

# ---------------------------
# Batch Upload
# ---------------------------
st.subheader("Upload CSV for Batch Predictions")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    batch = pd.read_csv(file)

    batch = clean_numeric_columns(batch)
    batch.fillna(batch.median(), inplace=True)

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