# app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from fpdf2 import FPDF
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Insurance KPI Dashboard", layout="wide")

model = joblib.load("best_model.pkl")
encoders = joblib.load("label_encoders.pkl")
features = joblib.load("feature_names.pkl")

with st.sidebar:
    page = option_menu("KPI Dashboard", [
        "Conversion Rate", "CLV Forecast", "CPA vs ROI",
        "Propensity Scorer", "Export Report"
    ])

# PAGE 1 – Your Chart
if page == "Conversion Rate":
    st.title("Conversion Rate by Channel")
    df = pd.DataFrame({
        "Channel": ["Agent", "Web", "Branch", "Call Center"],
        "Rate": [19.2, 11.8, 11.9, 10.9]
    })
    st.bar_chart(df.set_index("Channel"))
    st.markdown("""
    **Data-science boost**: Agent = **19.2%** (+34% vs 14.3% average)  
    *From your paper: "Agent-led sales hit 19.2% while Call Centers languish at 10.9%."*
    """)

# PAGE 4 – Propensity Scorer
elif page == "Propensity Scorer":
    st.title("Real-Time Propensity Scorer")
    HIGH_IMPACT = {
        "Customer Lifetime Value": 15000, "Monthly Premium Auto": 200,
        "Months Since Policy Inception": 6, "Sales Channel": "Agent",
        "Renew Offer Type": "Offer2", "Education": "Doctor",
        "EmploymentStatus": "Retired", "Gender": "F",
        "Location Code": "Suburban", "Coverage": "Premium",
        "Vehicle Class": "Luxury SUV"
    }

    with st.form("lead_form"):
        col1, col2 = st.columns(2)
        with col1:
            clv = st.number_input("CLV ($)", value=8000.0)
            premium = st.number_input("Premium ($)", value=100.0)
            months = st.number_input("Months", value=12)
        with col2:
            channel = st.selectbox("Channel", encoders["Sales Channel"].classes_)
            offer = st.selectbox("Offer", encoders["Renew Offer Type"].classes_)
            education = st.selectbox("Education", encoders["Education"].classes_)

        submitted = st.form_submit_button("SCORE LEAD")

    if st.button("Load 90%+ Lead"):
        for k, v in HIGH_IMPACT.items():
            st.session_state[k] = v
        st.rerun()

    if submitted:
        data = {
            "Customer Lifetime Value": clv,
            "Monthly Premium Auto": premium,
            "Months Since Policy Inception": months,
            "Sales Channel": channel,
            "Renew Offer Type": offer,
            "Education": education,
            "EmploymentStatus": "Retired",
            "Gender": "F",
            "Location Code": "Suburban",
            "Coverage": "Premium",
            "Vehicle Class": "Luxury SUV"
        }
        X = pd.DataFrame([data])
        for col in X.select_dtypes(include="object").columns:
            le = encoders[col]
            X[col] = le.transform(X[col])
        X = X[features]
        prob = model.predict_proba(X)[0, 1]
        st.success(f"**Propensity: {prob:.1%}**")
        explainer = shap.Explainer(model, X)
        shap_vals = explainer(X)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_vals[0], show=False)
        st.pyplot(fig)

# PAGE 5 – Your Full Article
elif page == "Export Report":
    st.title("Export Marketing Report")
    if st.button("Generate PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Data-Driven Marketing KPIs", ln=1, align="C")
        pdf.set_font("Arial", size=11)
        article = """
        In an era where every ad dollar is scrutinized, four KPIs—Conversion Rate, Customer
        Lifetime Value (CLV), Cost Per Acquisition (CPA), and Return on Investment (ROI)—are
        the North Star for profitable growth...

        This analysis of 9,000+ insurance prospects reveals a 14.3% overall conversion rate,
        yet Agent-led sales hit 19.2% while Call Centers languish at 10.9%.
        """
        pdf.multi_cell(0, 8, article)
        pdf.output("report.pdf")
        with open("report.pdf", "rb") as f:
            st.download_button("Download PDF", f, "Insurance_KPI_Report.pdf")