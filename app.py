# app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance Propensity Scorer", layout="wide")
st.title("Propensity-to-Buy Scorer")
st.markdown("### Enter lead details → Get **real-time** conversion probability + **SHAP explanation**")

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('propensity_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    features = joblib.load('feature_names.pkl')
    return model, encoders, features

model, encoders, features = load_artifacts()

# Input form
with st.form("lead_form"):
    col1, col2 = st.columns(2)
    with col1:
        clv = st.number_input("Customer Lifetime Value ($)", min_value=0.0, value=8000.0, step=100.0)
        premium = st.number_input("Monthly Premium Auto ($)", min_value=0.0, value=100.0, step=10.0)
        months = st.number_input("Months Since Policy Inception", min_value=0, value=12, step=1)
    with col2:
        channel = st.selectbox("Sales Channel", options=encoders['Sales Channel'].classes_)
        offer = st.selectbox("Renew Offer Type", options=encoders['Renew Offer Type'].classes_)
        education = st.selectbox("Education", options=encoders['Education'].classes_)

    col3, col4 = st.columns(2)
    with col3:
        employment = st.selectbox("Employment Status", options=encoders['EmploymentStatus'].classes_)
        gender = st.selectbox("Gender", options=encoders['Gender'].classes_)
        location = st.selectbox("Location Code", options=encoders['Location Code'].classes_)
    with col4:
        coverage = st.selectbox("Coverage", ["Basic", "Extended", "Premium"])
        vehicle = st.text_input("Vehicle Class", "Two-Door Car")

    submitted = st.form_submit_button("Score Lead")

# Predict
if submitted:
    with st.spinner("Scoring..."):
        data = {
            'Customer Lifetime Value': clv,
            'Monthly Premium Auto': premium,
            'Months_Since_Start': months,
            'Premium_Policy': 1 if coverage == 'Premium' else 0,
            'Luxury_Vehicle': 1 if 'Luxury' in vehicle else 0
        }
        for col in ['Sales Channel', 'Renew Offer Type', 'Education', 'EmploymentStatus', 'Gender', 'Location Code']:
            le = encoders[col]
            val = locals()[col.lower().replace(' ', '_')]
            data[col] = le.transform([val])[0] if val in le.classes_ else 0

        X = pd.DataFrame([data])[features]
        prob = model.predict_proba(X)[0, 1]

        st.success(f"**Propensity Score: {prob:.1%}**")
        st.metric("Recommended Action", "Route to Agent" if prob > 0.7 else "Web Funnel")

        # SHAP
        with st.spinner("Generating explanation..."):
            try:
                predict_fn = lambda x: model.predict_proba(x)[:, 1]
                explainer = shap.Explainer(predict_fn, X, feature_names=features)
                shap_values = explainer(X)
                fig, ax = plt.subplots()
                shap.waterfall_plot(shap_values[0], max_display=10, show=False)
                st.pyplot(fig)
            except:
                st.info("SHAP explanation skipped")