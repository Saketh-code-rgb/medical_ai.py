import streamlit as st
import pandas as pd
import job pickle # Assuming you save your model later

st.title("🏥 Chronic Disease Risk Predictor")
st.write("Enter patient data to assess risk levels.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)

if st.button("Analyze Risk"):
    # Simple logic for demonstration
    if glucose > 140 or bmi > 30:
        st.error("⚠️ Result: High Risk Detected. Consult a specialist.")
    else:
        st.success("✅ Result: Low Risk. Maintain healthy habits.")
