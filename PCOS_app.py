
import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(page_title="PCOS Risk Predictor", layout="centered", page_icon="🩺")

st.title(" Clinical PCOS Risk Predictor")
st.write(
    "This interactive web application uses a **Random Forest Machine Learning model** "
    "trained on clinical biomarkers to estimate the probability of Polycystic Ovary Syndrome (PCOS)."
)

st.sidebar.header("Patient Clinical Parameters")

# Sidebar inputs for clinical features
age = st.sidebar.slider("Age (years)", 18, 50, 28)
bmi = st.sidebar.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=23.5, step=0.1)
cycle_irregular = st.sidebar.selectbox("Irregular Menstrual Cycle?", ["No", "Yes"])
follicles_left = st.sidebar.slider("Follicle Count (Left Ovary)", 0, 30, 5)
follicles_right = st.sidebar.slider("Follicle Count (Right Ovary)", 0, 30, 6)
weight_gain = st.sidebar.selectbox("Involuntary Weight Gain?", ["No", "Yes"])
hirsutism = st.sidebar.selectbox("Excessive Hair Growth (Hirsutism)?", ["No", "Yes"])
amh = st.sidebar.number_input("Anti-Müllerian Hormone - AMH (ng/mL)", min_value=0.0, max_value=30.0, value=2.5, step=0.1)

# Encode inputs to match the model's numerical mapping
cycle_code = 4 if cycle_irregular == "Yes" else 2
weight_code = 1 if weight_gain == "Yes" else 0
hirsutism_code = 1 if hirsutism == "Yes" else 0

st.subheader("Diagnostic Evaluation")

# Prediction action
if st.button("Run Risk Prediction"):
    try:
        # Load the trained model
        model = joblib.load("models/rf_pcos_model.joblib")
        
        # Store user inputs in order
        patient_values = [
            age,
            bmi,
            cycle_code,
            follicles_left,
            follicles_right,
            weight_code,
            hirsutism_code,
            amh
        ]
        
        # Match exact feature names expected by the model
        patient_df = pd.DataFrame([patient_values], columns=model.feature_names_in_)
        
        # Predict probability
        probabilities = model.predict_proba(patient_df)
        pcos_risk = probabilities[0][1]
        
        st.markdown("---")
        if pcos_risk >= 0.50:
            st.error(f" **High PCOS Risk:** {pcos_risk:.1%}")
        else:
            st.success(f" **Low PCOS Risk:** {pcos_risk:.1%}")
            
        st.info("ℹ️ *Disclaimer: This app is strictly designed for educational and technical demonstration purposes. It does not constitute formal medical diagnosis or advice.*")

    except FileNotFoundError:
        st.error(" Model file 'models/rf_pcos_model.joblib' not found. Please run 'python pcos.py' first.")
    except Exception as e:
        st.error(f" An unexpected error occurred: {e}")
