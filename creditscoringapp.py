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
    "NumberOfDependents"
]

# ---------------------------
# ULTRA-ROBUST CLEANING PIPELINE
# ---------------------------
def to_model_ready(df):
    df = df.copy()

    # remove weird string artifacts like [5E-1]
    df = df.replace(r"[\[\]'\"]", "", regex=True)

    # force numeric conversion (kills all bad strings)
    df = df.apply(pd.to_numeric, errors="coerce")

    # fill missing values safely
    df = df.fillna(df.median(numeric_only=True))

    # final strict type
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
    raw_df = pd.read_csv(BytesIO(obj["Body"].read()))

    raw_df = to_model_ready(raw_df)

    data_df = raw_df[FEATURE_COLUMNS]

    st.success("Data loaded and fully cleaned")

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
# SAFE MODEL INPUT
# ---------------------------
X = data_df.values.astype(np.float32)

scaled_X = scaler.transform(X)

# ---------------------------
# PREDICTIONS
# ---------------------------
st.subheader("Predictions")

data_df["LogReg_Prob"] = logreg_model.predict_proba(scaled_X)[:, 1]
data_df["XGB_Prob"] = xgb_model.predict_proba(X)[:, 1]

st.dataframe(data_df)
st.download_button("Download Predictions", data_df.to_csv(index=False), "predictions.csv")

# ---------------------------
# SHAP EXPLANATION (FULLY FIXED)
# ---------------------------
st.subheader("Business Interpretation (XGBoost)")

try:
    clean_X = data_df[FEATURE_COLUMNS].values.astype(np.float32)

    sample_row = np.median(clean_X, axis=0).reshape(1, -1)
    background = clean_X[np.random.choice(clean_X.shape[0], min(50, clean_X.shape[0]), replace=False)]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(sample_row)

    # Plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # Feature importance
    feature_names = FEATURE_COLUMNS

    feature_impact = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": shap_values.values[0]
    }).sort_values(by="SHAP_Value", key=abs, ascending=False)

    st.markdown("### Top 3 Features Driving Risk")

    for _, row in feature_impact.head(3).iterrows():
        direction = "increases" if row["SHAP_Value"] > 0 else "decreases"
        st.write(
            f"- {row['Feature']} {direction} default risk (impact: {row['SHAP_Value']:.4f})"
        )

except Exception as e:
    st.error(f"SHAP failed: {e}")

# ---------------------------
# BATCH UPLOAD
# ---------------------------
st.subheader("Upload CSV for Batch Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    batch = pd.read_csv(file)
    batch = to_model_ready(batch)
    batch = batch[FEATURE_COLUMNS]

    X_batch = batch.values.astype(np.float32)
    X_batch_scaled = scaler.transform(X_batch)

    batch["LogReg_Prob"] = logreg_model.predict_proba(X_batch_scaled)[:, 1]
    batch["XGB_Prob"] = xgb_model.predict_proba(X_batch)[:, 1]

    st.dataframe(batch)
    st.download_button("Download Batch Results", batch.to_csv(index=False), "batch_predictions.csv")