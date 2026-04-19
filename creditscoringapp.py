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
Data is loaded from Cloudflare R2 and can also be uploaded manually.
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
# SAFE CLEANING FUNCTION (FIXED)
# ---------------------------
def clean_numeric_columns(df):
    df = df.copy()

    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[\[\]'\"]", "", regex=True)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ---------------------------
# LOAD DATA FROM R2
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

    # ---------------------------
    # Feature Engineering FIRST
    # ---------------------------
    data_df["TotalPastDue"] = (
        data_df["NumberOfTime30-59DaysPastDueNotWorse"] +
        data_df["NumberOfTime60-89DaysPastDueNotWorse"] +
        data_df["NumberOfTimes90DaysLate"]
    )

    data_df["DebtPerIncome"] = data_df["DebtRatio"] * data_df["MonthlyIncome"]

    # ---------------------------
    # CLEAN DATA
    # ---------------------------
    data_df = clean_numeric_columns(data_df)

    # Safe median fill
    data_df.fillna(data_df.median(numeric_only=True), inplace=True)

    data_df = data_df[FEATURE_COLUMNS]

    st.success("Dataset loaded successfully")

except Exception as e:
    st.warning(f"R2 load failed: {e}")
    st.stop()

# ---------------------------
# LOAD MODELS
# ---------------------------
try:
    logreg_model = joblib.load("models/logreg_v2.pkl")
    xgb_model = joblib.load("models/xgb_best.pkl")
    scaler = joblib.load("models/scaler_v2.pkl")

    st.success("Models loaded successfully")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------------------------
# PREDICTIONS
# ---------------------------
st.subheader("Predictions on Cloudflare Dataset")

features_df = data_df.copy()

# Logistic Regression (scaled)
scaled_data = scaler.transform(features_df)
data_df["LogReg_Prob"] = logreg_model.predict_proba(scaled_data)[:, 1]

# XGBoost (raw cleaned features)
data_df["XGB_Prob"] = xgb_model.predict_proba(features_df)[:, 1]

st.dataframe(data_df)
st.download_button("Download Predictions", data_df.to_csv(index=False), "predictions.csv")

# ---------------------------
# SHAP INTERPRETATION (FIXED)
# ---------------------------
st.subheader("Business Interpretation (XGBoost)")

try:
    sample_row = data_df[FEATURE_COLUMNS].median().to_frame().T

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(sample_row)

    # Plot
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # FIXED SHAP extraction
    shap_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "SHAP_Value": shap_values.values[0]
    })

    shap_df["abs"] = shap_df["SHAP_Value"].abs()
    shap_df = shap_df.sort_values("abs", ascending=False).drop(columns=["abs"])

    st.markdown("**Top 3 Features Driving Prediction:**")

    for _, row in shap_df.head(3).iterrows():
        direction = "increases" if row["SHAP_Value"] > 0 else "decreases"

        st.write(
            f"- {row['Feature']} {direction} risk "
            f"(impact: {row['SHAP_Value']:.3f})"
        )

except Exception as e:
    st.warning(f"Business Interpretation failed: {e}")

# ---------------------------
# BATCH UPLOAD
# ---------------------------
st.subheader("Upload CSV for Batch Prediction")

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
    st.download_button("Download Predictions", batch.to_csv(index=False), "batch_predictions.csv")