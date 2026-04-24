import streamlit as st
import pandas as pd
import joblib

# =========================
# Load Model
# =========================
model = joblib.load("xgboost_salary_pipeline.pkl")
threshold = joblib.load("xgboost_threshold_0_40.pkl")

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Salary Predictor", layout="centered")

# =========================
# Title
# =========================
st.title("💼 Salary Prediction App")
st.write("Predict whether income is >50K or <=50K")

st.markdown("---")

# =========================
# Inputs (Main Page)
# =========================
st.subheader("👤 Personal Information")

age = st.slider("Age", 18, 80, 30)
gender = st.selectbox("Gender", ["Male", "Female"])

marital_status = st.selectbox("Marital Status", [
    "Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"
])

relationship = st.selectbox("Relationship", [
    "Husband", "Wife", "Not-in-family", "Unmarried", "Own-child"
])

race = st.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander",
    "Amer-Indian-Eskimo", "Other"
])

st.markdown("---")

st.subheader("🎓 Work & Education")

education_years = st.slider("Education Years", 1, 20, 10)

employment_type = st.selectbox("Employment Type", [
    "Private", "Self-emp-not-inc", "Government", "Other"
])

job_role = st.selectbox("Job Role", [
    "Exec-managerial", "Tech-support", "Other-service",
    "Sales", "Craft-repair"
])

hours = st.slider("Weekly Work Hours", 1, 80, 40)

st.markdown("---")

st.subheader("💰 Financial Information")

capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

country = st.selectbox("Country", [
    "United-States", "India", "Mexico", "Other"
])

st.markdown("---")

# =========================
# Prediction
# =========================
if st.button("🔮 Predict Salary"):

    data = pd.DataFrame([{
        "Age": age,
        "Employment_Type": employment_type,
        "Education_Years": education_years,
        "Marital_Status": marital_status,
        "Job_Role": job_role,
        "Relationship": relationship,
        "Race": race,
        "Gender": 1 if gender == "Male" else 0,
        "Capital_Gain": capital_gain,
        "Capital_Loss": capital_loss,
        "Weekly_Work_Hours": hours,
        "Country": country,
        "Education_Group": "Undergraduate"
    }])

    proba_raw = model.predict_proba(data)[:, 1][0]
    proba = max(min(proba_raw, 0.95), 0.05)

    pred = 1 if proba >= threshold else 0

    st.markdown("---")

    if pred == 1:
        st.success("💰 Income > 50K")
    else:
        st.error("📉 Income <= 50K")

    st.write("### Model Confidence (approximate)")
    st.progress(float(proba))
    st.write(f"Score: {proba:.2f}")

    if proba > 0.75:
        st.success("High confidence prediction")
    elif proba > 0.5:
        st.info("Moderate confidence prediction")
    else:
        st.warning("Low confidence prediction")
     # =========================
    # Final Input Summary
    # =========================
    st.markdown("### 📋 Entered Details")

    st.write(f"Age: {age}")
    st.write(f"Gender: {gender}")
    st.write(f"Marital Status: {marital_status}")
    st.write(f"Relationship: {relationship}")
    st.write(f"Race: {race}")
    st.write(f"Education Years: {education_years}")
    st.write(f"Employment Type: {employment_type}")
    st.write(f"Job Role: {job_role}")
    st.write(f"Weekly Hours: {hours}")
    st.write(f"Capital Gain: {capital_gain}")
    st.write(f"Capital Loss: {capital_loss}")
    st.write(f"Country: {country}")
