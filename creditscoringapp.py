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
Predictions are automatically run on the dataset loaded from Cloudflare R2. You can also upload your own CSV for batch predictions.
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
# CLEANING (FIXED)
# ---------------------------
def clean_numeric_columns(df):
    def safe_convert(x):
        try:
            if isinstance(x, str):
                x = x.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
                return float(x)
            return float(x)
        except:
            return np.nan

    df = df.applymap(safe_convert)

    # force full numeric conversion (CRITICAL FIX)
    df = df.apply(pd.to_numeric, errors="coerce")

    return df

# ---------------------------
# INIT
# ---------------------------
batch = None
data_df = None

# ---------------------------
# LOAD DATA
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

    # CLEAN (FIXED)
    data_df = clean_numeric_columns(data_df)
    data_df.fillna(data_df.median(numeric_only=True), inplace=True)

    # FEATURE ENGINEERING
    data_df['TotalPastDue'] = (
        data_df['NumberOfTime30-59DaysPastDueNotWorse'] +
        data_df['NumberOfTime60-89DaysPastDueNotWorse'] +
        data_df['NumberOfTimes90DaysLate']
    )

    data_df['DebtPerIncome'] = data_df['DebtRatio'] * data_df['MonthlyIncome']

    data_df = data_df[FEATURE_COLUMNS]

    st.success("Dataset loaded successfully")

except Exception as e:
    st.warning(f"Could not load dataset: {e}")
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
    st.error(f"Model load error: {e}")
    st.stop()

# ---------------------------
# PREDICTIONS
# ---------------------------
st.subheader("Predictions on Cloudflare Dataset")

features_df = data_df[FEATURE_COLUMNS].copy()

# FINAL SAFETY CLEAN (IMPORTANT FIX)
features_df = features_df.apply(pd.to_numeric, errors="coerce")
features_df.fillna(features_df.median(), inplace=True)

scaled_data = scaler.transform(features_df)

data_df["LogReg_Prob"] = logreg_model.predict_proba(scaled_data)[:, 1]
data_df["XGB_Prob"] = xgb_model.predict_proba(features_df)[:, 1]

st.dataframe(data_df)
st.download_button("Download Predictions", data_df.to_csv(index=False), "predictions.csv")

# ---------------------------
# SHAP SECTION (FIXED)
# ---------------------------
st.subheader("Business Interpretation (XGBoost)")

try:
    if batch is not None and all(col in batch.columns for col in FEATURE_COLUMNS):
        sample_row = batch[FEATURE_COLUMNS].iloc[[0]]
        background = batch[FEATURE_COLUMNS].sample(min(50, len(batch)))

    else:
        sample_row = data_df[FEATURE_COLUMNS].iloc[[0]]
        background = data_df[FEATURE_COLUMNS].sample(min(50, len(data_df)))

    # FINAL SAFETY CLEAN BEFORE SHAP (CRITICAL FIX)
    sample_row = sample_row.apply(pd.to_numeric, errors="coerce")
    background = background.apply(pd.to_numeric, errors="coerce")

    sample_row.fillna(sample_row.median(numeric_only=True), inplace=True)
    background.fillna(background.median(numeric_only=True), inplace=True)

    # SHAP
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(sample_row)

    # PLOT
    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=sample_row.iloc[0]
        ),
        show=False
    )
    st.pyplot(fig)

    # FEATURE IMPORTANCE
    feature_impact = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "SHAP_Value": shap_values[0]
    }).sort_values(by="SHAP_Value", key=abs, ascending=False)

    st.markdown("**Top 3 features influencing the XGBoost prediction:**")

    for i, row in feature_impact.head(3).iterrows():
        direction = "increases" if row["SHAP_Value"] > 0 else "decreases"

        st.write(
            f"- {row['Feature']} {direction} the likelihood of delinquency "
            f"(impact: {row['SHAP_Value']:.2f})"
        )

except Exception as e:
    st.warning(f"Business Interpretation failed: {e}")

# ---------------------------
# UPLOAD SECTION
# ---------------------------
st.subheader("Upload Your CSV for Batch Predictions")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    batch = pd.read_csv(file)

    batch = clean_numeric_columns(batch)
    batch.fillna(batch.median(numeric_only=True), inplace=True)

    batch['TotalPastDue'] = (
        batch['NumberOfTime30-59DaysPastDueNotWorse'] +
        batch['NumberOfTime60-89DaysPastDueNotWorse'] +
        batch['NumberOfTimes90DaysLate']
    )

    batch['DebtPerIncome'] = batch['DebtRatio'] * batch['MonthlyIncome']

    batch_features = batch[FEATURE_COLUMNS]

    batch_features = batch_features.apply(pd.to_numeric, errors="coerce")
    batch_features.fillna(batch_features.median(), inplace=True)

    batch_scaled = scaler.transform(batch_features)

    batch["LogReg_Prob"] = logreg_model.predict_proba(batch_scaled)[:, 1]
    batch["XGB_Prob"] = xgb_model.predict_proba(batch_features)[:, 1]

    st.dataframe(batch)
    st.download_button("Download Predictions", batch.to_csv(index=False), "predictions.csv")