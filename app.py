# Employee Attrition Prediction – Streamlit App

import streamlit as st
import pandas as pd
import joblib

# IMPORTANT: import custom transformer
from feature_engineering import FeatureEngineering

# Page Configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# Load Trained Pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("attrition_pipeline.pkl")

pipeline = load_pipeline()

# Title
st.title("📊 Employee Attrition Prediction Dashboard")
st.markdown("Predict whether an employee is likely to **leave or stay** based on key factors.")

st.divider()

# User Inputs
st.subheader("🔢 Enter Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", 18, 60, 30)
    DistanceFromHome = st.slider("Distance From Home", 1, 30, 5)
    MonthlyIncome = st.number_input("Monthly Income", 1000, step=500)
    PercentSalaryHike = st.slider("Percent Salary Hike", 0, 30, 10)

with col2:
    JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 2)
    EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 2)
    RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 2)
    WorkLifeBalance = st.slider("Work Life Balance", 1, 4, 2)

with col3:
    JobInvolvement = st.slider("Job Involvement", 1, 4, 2)
    YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
    YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    YearsWithCurrManager = st.slider("Years With Current Manager", 0, 15, 2)

st.divider()

# Build FULL input DataFrame 
# Get exact feature list used during training
model_features = pipeline.named_steps["scaler"].feature_names_in_

# Create empty DataFrame with ALL required features
input_df = pd.DataFrame(
    data=[[0] * len(model_features)],
    columns=model_features
)

# User-provided values (subset of features)
user_inputs = {
    "Age": Age,
    "DistanceFromHome": DistanceFromHome,
    "MonthlyIncome": MonthlyIncome,
    "PercentSalaryHike": PercentSalaryHike,
    "JobSatisfaction": JobSatisfaction,
    "EnvironmentSatisfaction": EnvironmentSatisfaction,
    "RelationshipSatisfaction": RelationshipSatisfaction,
    "WorkLifeBalance": WorkLifeBalance,
    "JobInvolvement": JobInvolvement,
    "YearsAtCompany": YearsAtCompany,
    "YearsSinceLastPromotion": YearsSinceLastPromotion,
    "YearsWithCurrManager": YearsWithCurrManager,
}

# Fill user inputs into full feature set
for col, val in user_inputs.items():
    if col in input_df.columns:
        input_df[col] = val

# Prediction
if st.button("🔍 Predict Attrition", use_container_width=True):

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error("⚠️ **High Risk of Attrition**")
        st.metric("Probability of Leaving", f"{probability:.2%}")
    else:
        st.success("✅ **Low Risk – Employee Likely to Stay**")
        st.metric("Probability of Staying", f"{(1 - probability):.2%}")

# Footer
st.divider()
st.caption("🚀 Built with Scikit-learn Pipeline & Streamlit")
