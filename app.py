# ==============================================================
# STREAMLIT PROPENSITY SCORING APP – NO API NEEDED
# ==============================================================

import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ------------------- Load Model & Artifacts -------------------
@st.cache_resource
def load_model():
    model = joblib.load('propensity_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    features = joblib.load('feature_names.pkl')
    return model, encoders, features

model, encoders, features = load_model()

# ------------------- Streamlit UI -------------------
st.title("Propensity-to-Buy Scorer")
st.markdown("Enter lead details → Get real-time conversion probability")

col1, col2 = st.columns(2)

with col1:
    clv = st.number_input("Customer Lifetime Value ($)", min_value=0.0, value=8000.0)
    premium = st.number_input("Monthly Premium Auto ($)", min_value=0.0, value=100.0)
    months = st.number_input("Months Since Policy Inception", min_value=0, value=12)

with col2:
    channel = st.selectbox("Sales Channel", encoders['Sales Channel'].classes_)
    offer = st.selectbox("Renew Offer Type", encoders['Renew Offer Type'].classes_)
    education = st.selectbox("Education", encoders['Education'].classes_)

col3, col4 = st.columns(2)
with col3:
    employment = st.selectbox("Employment Status", encoders['EmploymentStatus'].classes_)
    gender = st.selectbox("Gender", encoders['Gender'].classes_)
    location = st.selectbox("Location Code", encoders['Location Code'].classes_)

with col4:
    coverage = st.selectbox("Coverage", ["Basic", "Extended", "Premium"])
    vehicle = st.text_input("Vehicle Class (e.g., Luxury SUV)", "Two-Door Car")

# ------------------- Predict -------------------
if st.button("Score This Lead"):
    # Build feature dict
    data = {
        'Customer Lifetime Value': clv,
        'Monthly Premium Auto': premium,
        'Months_Since_Start': months,
        'Premium_Policy': 1 if coverage == 'Premium' else 0,
        'Luxury_Vehicle': 1 if 'Luxury' in vehicle else 0
    }
    # Encode categoricals
    for col in ['Sales Channel', 'Renew Offer Type', 'Education', 'EmploymentStatus', 'Gender', 'Location Code']:
        le = encoders[col]
        val = locals()[col.lower().replace(' ', '_')]
        data[col] = le.transform([val])[0]

    X = pd.DataFrame([data])[features]
    prob = model.predict_proba(X)[0, 1]

    st.metric("Propensity Score", f"{prob:.1%}")
    st.write(f"**Action:** {'Route to Agent' if prob > 0.7 else 'Web Funnel'}")

    # ------------------- SHAP Explanation -------------------
    with st.spinner("Generating explanation..."):
        booster = model.get_booster()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X)

        fig, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X.iloc[0]), show=False)
        st.pyplot(fig)