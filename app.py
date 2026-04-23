import streamlit as st
import pandas as pd
import joblib

# =========================
# Load Model
# =========================
model = joblib.load("salary_model_pipeline.pkl")
threshold = joblib.load("salary_threshold.pkl")

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
# Sidebar Inputs
# =========================
st.sidebar.header("Enter Details")

age = st.sidebar.slider("Age", 18, 80, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"
])

relationship = st.sidebar.selectbox("Relationship", [
    "Husband", "Wife", "Not-in-family", "Unmarried", "Own-child"
])

race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander",
    "Amer-Indian-Eskimo", "Other"
])

education_years = st.sidebar.slider("Education Years", 1, 20, 10)

employment_type = st.sidebar.selectbox("Employment Type", [
    "Private", "Self-emp-not-inc", "Government", "Other"
])

job_role = st.sidebar.selectbox("Job Role", [
    "Exec-managerial", "Tech-support", "Other-service",
    "Sales", "Craft-repair"
])

hours = st.sidebar.slider("Weekly Work Hours", 1, 80, 40)

capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)

country = st.sidebar.selectbox("Country", [
    "United-States", "India", "Mexico", "Other"
])

# =========================
# Prediction
# =========================
if st.button("Predict"):

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

    # Raw probability
    proba_raw = model.predict_proba(data)[:, 1][0]

    # 🔧 Smooth extreme values (UI-level only)
    proba = max(min(proba_raw, 0.95), 0.05)

    pred = 1 if proba >= threshold else 0

    st.markdown("---")

    # =========================
    # Result
    # =========================
    if pred == 1:
        st.success("💰 Income > 50K")
    else:
        st.error("📉 Income <= 50K")

    
    st.write("### Model Confidence (approximate)")

    st.progress(float(proba))
    st.write(f"Score: {proba:.2f}")

    # Explanation (important)
    st.caption("Note: This score reflects model confidence and is not a calibrated probability.")

    # Insight message
    if proba > 0.75:
        st.success("High confidence prediction")
    elif proba > 0.5:
        st.info("Moderate confidence prediction")
    else:
        st.warning("Low confidence prediction")