"""
Medical Insurance Cost Prediction — Phase 7 Application
Run locally with:  streamlit run app.py
"""

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💰", layout="centered")

# ---------------------------------------------------------------------------
# Load trained model (produced by insurance_prediction.ipynb, Phases 1-6)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("insurance_model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_model()
model = bundle["model"]
feature_columns = bundle["feature_columns"]
metrics = bundle["metrics"]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("💰 Medical Insurance Cost Predictor")
st.write(
    "Enter a customer's details below to estimate their annual medical insurance charges, "
    "based on a Linear Regression model trained on the Medical Cost Personal Dataset."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)

    with col2:
        sex = st.selectbox("Sex", ["female", "male"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    submitted = st.form_submit_button("Predict Insurance Cost")

if submitted:
    # Build a single-row dataframe matching the training encoding exactly
    row = {
        "age": age,
        "sex": 1 if sex == "male" else 0,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }
    input_df = pd.DataFrame([row])[feature_columns]

    prediction = model.predict(input_df)[0]
    prediction = max(prediction, 0)  # charges can't be negative

    st.success(f"### Estimated Annual Insurance Cost: **${prediction:,.2f}**")

    st.caption(
        f"Model performance on test data — MAE: ${metrics['MAE']:,.0f} | "
        f"RMSE: ${metrics['RMSE']:,.0f} | R²: {metrics['R2']:.3f}"
    )

st.divider()
st.caption(
    "Built with Python, Pandas, Scikit-learn & Streamlit — "
    "trained on the Kaggle Medical Cost Personal Dataset."
)
