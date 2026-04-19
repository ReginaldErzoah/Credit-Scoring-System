# Credit Scoring & Loan Decision System

## Project Overview

This is an interactive **Streamlit dashboard** for **credit risk assessment**.
It predicts the likelihood of a customer defaulting on a loan using **Logistic Regression** and **XGBoost**, and provides explainable insights using **SHAP**.

The app provides:

* **Automatic predictions** on the dataset stored in Cloudflare R2
* **Batch CSV upload** for custom predictions
* **SHAP-based interpretability** showing top features influencing predictions
* **Pre-trained ML models** and feature scaler for fast inference

The project demonstrates:

* Python for **data preprocessing, feature engineering, and modeling**
* Pandas & NumPy for **data wrangling**
* Logistic Regression & XGBoost for **classification and probability predictions**
* Streamlit & Matplotlib for **interactive dashboards and visualizations**
* Cloudflare R2 for **secure cloud-hosted dataset storage**
* Docker for **containerized deployment**

---

## Live Demo

Check out the live dashboard here: [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-scoring-system1.streamlit.app/)

---

## Use Case

This app is ideal for:

* **Banks and lenders** assessing loan applications
* **Data analysts** exploring credit risk patterns
* **Business owners** monitoring financial risk exposure

---

## Features

**Automated Credit Risk Prediction**

* Predict default probability for each customer
* Generate actionable insights for loan approval decisions
* SHAP feature explanation for top 3 predictors

**Batch Predictions**

* Upload custom CSVs to score multiple customers at once
* Download predictions for further analysis

**Interactive Visualizations**

* SHAP waterfall plots showing feature impact
* Top features increasing or decreasing default risk

**Pre-trained & Optimized Models**

* Logistic Regression & XGBoost models ready for production

* Logistic Regression and XGBoost models were tuned using hyperparameter optimization.
  
**Cross-Validation**
  
* Model performance was validated using k-fold cross-validation to ensure reliability and reduce overfitting.

**Pre-trained Scaler**

* Features were scaled using StandardScaler to standardize input for Logistic Regression and XGBoost.
---

## Dataset

The dataset consists of historical credit information, including:

* `RevolvingUtilizationOfUnsecuredLines`
* `Age`
* `DebtRatio`
* `MonthlyIncome`
* `NumberOfTimes90DaysLate`
* `NumberRealEstateLoansOrLines`
* `NumberOfDependents`
* And more engineered features like `TotalPastDue` and `DebtPerIncome`

Data is hosted in **Cloudflare R2** for secure, scalable storage.
The app accesses the dataset directly from the cloud bucket.

---

## Tech Stack

| Category          | Tools                                       |
| ----------------- | ------------------------------------------- |
| Programming       | Python                                      |
| Data Handling     | Pandas, NumPy                               |
| Machine Learning  | Scikit-learn (Logistic Regression, XGBoost, StandardScaler, Hyperparameter Tuning, Cross-Validation) |
| Explainability    | SHAP                                        |
| Visualization     | Matplotlib, Streamlit                       |
| Web App           | Streamlit                                   |
| Deployment        | Streamlit Cloud, Docker                                      |
| Model Persistence | joblib                                      |
| Notebook Analysis | Jupyter Notebook                            |
| Cloud Storage     | Cloudflare R2 (S3-compatible)               |

---

## Architecture & Workflow

```
Customer Credit Data → Cloudflare R2 → Streamlit App → Predictions + SHAP Interpretations
           ↑
           └── Pre-trained ML Models (LogReg + XGBoost) + Scaler
```

1. **Data Load** – Dataset accessed from Cloudflare R2
2. **Feature Engineering** – Compute `TotalPastDue` and `DebtPerIncome`
3. **Prediction** – Score default probability using ML models
4. **Interpretation** – SHAP waterfall plots identify top predictors
5. **Batch Upload** – Option to upload custom CSVs for predictions

---

## Update & Version Log

* **Version 1.0** (March 2026): Initial release with automated predictions, SHAP interpretability, batch uploads, and containerized deployment using Docker

