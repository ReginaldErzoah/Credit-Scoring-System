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
It supports batch predictions and SHAP-based explainability.
""")

# ---------------------------
# Feature Columns (CLEAN — SAME AS TRAINING)
# ---------------------------
FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
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
                        .replace('"', ''))
        if isinstance(x, str) else x
    )

# ---------------------------
# Initialize
# ---------------------------
batch = None
data_df = None

# ---------------------------
# Load Data from Cloudflare R2
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
    data_df.fillna(data_df.median(numeric_only=True), inplace=True)

    data_df = data_df[FEATURE_COLUMNS]

    st.success("Dataset loaded successfully from Cloudflare R2")

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
# SHAP EXPLANATION (FIXED)
# ---------------------------
st.subheader("Business Interpretation (XGBoost - SHAP)")

try:
    # Use real sample (NOT median)
    sample_row = data_df[FEATURE_COLUMNS].sample(1)

    # Background data for stability
    background = data_df[FEATURE_COLUMNS].sample(min(100, len(data_df)))

    # Correct SHAP for tree models
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample_row)

    # ---------------------------
    # Waterfall Plot
    # ---------------------------
    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=sample_row.iloc[0].values,
            feature_names=FEATURE_COLUMNS
        ),
        show=False
    )
    st.pyplot(fig)

    # ---------------------------
    # Top Features
    # ---------------------------
    feature_impact = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "SHAP_Value": shap_values[0]
    }).sort_values(by="SHAP_Value", key=abs, ascending=False)

    st.markdown("### Top 3 Drivers of Prediction")

    for _, row in feature_impact.head(3).iterrows():
        direction = "increases" if row["SHAP_Value"] > 0 else "decreases"
        st.write(
            f"- {row['Feature']} {direction} default risk (impact: {row['SHAP_Value']:.3f})"
        )

except Exception as e:
    st.warning(f"SHAP explanation failed: {e}")

# ---------------------------
# Batch Upload
# ---------------------------
st.subheader("Upload CSV for Batch Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    batch = pd.read_csv(file)

    batch = clean_numeric_columns(batch)
    batch.fillna(batch.median(numeric_only=True), inplace=True)

    batch = batch[FEATURE_COLUMNS]

    scaled_batch = scaler.transform(batch)

    batch["LogReg_Prob"] = logreg_model.predict_proba(scaled_batch)[:, 1]
    batch["XGB_Prob"] = xgb_model.predict_proba(batch)[:, 1]

    st.dataframe(batch)

    st.download_button(
        "Download Batch Predictions",
        batch.to_csv(index=False),
        "batch_predictions.csv"
    )