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
Data is loaded from Cloudflare R2, cleaned, and scored automatically.
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
# CLEANING FUNCTION (FIXED + SAFE)
# ---------------------------
def clean_numeric_columns(df):
    df = df.copy()

    df = df.replace(r"[\[\]'\"]", "", regex=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median(numeric_only=True))

    return df.astype(np.float32)

# ---------------------------
# LOAD DATA FROM R2
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

    data_df = clean_numeric_columns(data_df)
    data_df = data_df[FEATURE_COLUMNS]

    st.success("Dataset loaded successfully")

except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

# ---------------------------
# LOAD MODELS
# ---------------------------
try:
    logreg_model = joblib.load("models/logreg_v3.pkl")
    xgb_model = joblib.load("models/xgb_best.pkl")
    scaler = joblib.load("models/scaler_v3.pkl")

    st.success("Models loaded successfully")

except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# ---------------------------
# PREDICTIONS
# ---------------------------
st.subheader("Predictions")

X = data_df.values.astype(np.float32)
X_scaled = scaler.transform(X)

data_df["LogReg_Prob"] = logreg_model.predict_proba(X_scaled)[:, 1]
data_df["XGB_Prob"] = xgb_model.predict_proba(X)[:, 1]

st.dataframe(data_df)
st.download_button("Download Predictions", data_df.to_csv(index=False), "predictions.csv")

# ---------------------------
# SHAP INTERPRETATION (FIXED FINAL VERSION)
# ---------------------------
st.subheader("Business Interpretation (XGBoost)")

try:
    X_clean = data_df.values.astype(np.float32)

    sample_row = np.median(X_clean, axis=0).reshape(1, -1)
    background = X_clean[np.random.choice(X_clean.shape[0], min(50, X_clean.shape[0]), replace=False)]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(sample_row)

    # FIXED WATERFALL
    shap.plots.waterfall(shap_values[0])
    st.pyplot(plt.gcf())

    feature_impact = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "SHAP_Value": shap_values.values[0]
    }).sort_values(by="SHAP_Value", key=abs, ascending=False)

    st.markdown("### Top 3 Features Driving Default Risk")

    for _, row in feature_impact.head(3).iterrows():
        direction = "increases default risk" if row["SHAP_Value"] > 0 else "decreases default risk"
        st.write(f"- {row['Feature']} {direction} ({row['SHAP_Value']:.4f})")

except Exception as e:
    st.error(f"SHAP failed: {e}")

# ---------------------------
# BATCH UPLOAD
# ---------------------------
st.subheader("Upload CSV for Batch Predictions")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    batch = pd.read_csv(file)

    batch = clean_numeric_columns(batch)
    batch = batch[FEATURE_COLUMNS]

    X_batch = batch.values.astype(np.float32)
    X_batch_scaled = scaler.transform(X_batch)

    batch["LogReg_Prob"] = logreg_model.predict_proba(X_batch_scaled)[:, 1]
    batch["XGB_Prob"] = xgb_model.predict_proba(X_batch)[:, 1]

    st.dataframe(batch)
    st.download_button("Download Batch Results", batch.to_csv(index=False), "batch_predictions.csv")