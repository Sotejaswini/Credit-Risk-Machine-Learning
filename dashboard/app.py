
"""Streamlit dashboard for the educational credit-risk model."""
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.feature_engineering import engineer_features
from src.credit_scoring import pd_to_score, assign_risk_band

st.set_page_config(page_title="Credit Risk Analytics", page_icon="📊", layout="wide")
st.title("Credit Risk Analytics Platform")
st.caption("Educational PD model demonstration — not a lending decision system.")

model_options = {
    "Logistic Regression": ROOT / "models" / "logistic_regression.joblib",
    "Random Forest": ROOT / "models" / "random_forest.joblib",
    "XGBoost": ROOT / "models" / "xgboost.joblib"
}
available = {k: v for k, v in model_options.items() if v.exists()}
if not available:
    st.error("No trained model found. Run notebooks/03_Model_Training.ipynb first.")
    st.stop()

model_name = st.sidebar.selectbox("Select model", list(available))
model = joblib.load(available[model_name])

st.subheader("1. Score a customer record")
st.write("Enter the original dataset features. Feature engineering is applied automatically.")
raw_features = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]
defaults = {
    "LIMIT_BAL": 150000, "SEX": 2, "EDUCATION": 2,
    "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 50000, "BILL_AMT2": 45000, "BILL_AMT3": 40000,
    "BILL_AMT4": 35000, "BILL_AMT5": 30000, "BILL_AMT6": 25000,
    "PAY_AMT1": 5000, "PAY_AMT2": 5000, "PAY_AMT3": 5000,
    "PAY_AMT4": 5000, "PAY_AMT5": 5000, "PAY_AMT6": 5000
}
with st.form("customer_form"):
    cols = st.columns(3)
    values = {}
    for i, feature in enumerate(raw_features):
        with cols[i % 3]:
            values[feature] = st.number_input(
                feature, value=float(defaults[feature]), step=1.0
            )
    submitted = st.form_submit_button("Calculate estimated PD")

if submitted:
    input_df = pd.DataFrame([values])
    engineered = engineer_features(input_df)
    pd_value = float(model.predict_proba(engineered)[:, 1][0])
    score = int(pd_to_score([pd_value])[0])
    band = str(assign_risk_band([pd_value])[0])
    a, b, c = st.columns(3)
    a.metric("Estimated probability of default", f"{pd_value:.2%}")
    b.metric("Illustrative credit score", score)
    c.metric("Illustrative risk band", band)
    st.warning("This is a demonstration on historical Taiwan credit-card data. It must not be used for actual credit decisions.")

st.divider()
st.subheader("2. Model comparison")
comparison_path = ROOT / "reports" / "training_model_comparison.csv"
if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison, use_container_width=True)
    st.bar_chart(comparison.set_index("model")["roc_auc"])
else:
    st.info("Run model training to generate the model comparison report.")

st.subheader("3. Project scope")
st.markdown("""
- Probability of Default (PD) estimation
- Logistic Regression, Random Forest and XGBoost
- ROC-AUC, Gini, KS and Brier score
- Illustrative credit score and risk bands
- PSI population drift monitoring
- Educational IFRS 9 staging and ECL examples
""")
