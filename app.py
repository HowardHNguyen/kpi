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
        "Conversion Rate", "Propensity Scorer", "Export Report"
    ])

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
    st.image("https://i.imgur.com/5vXjK9P.png", caption="Conversion Rate by Sales Channel")

elif page == "Propensity Scorer":
    st.title("Real-Time Propensity Scorer")
    with st.form("lead_form"):
        col1, col2 = st.columns(2)
        with col1:
            clv = st.number_input("CLV ($)", value=8000.0)
            premium = st.number_input("Premium ($)", value=100.0)
        with col2:
            channel = st.selectbox("Channel", encoders["Sales Channel"].classes_)
            offer = st.selectbox("Offer", encoders["Renew Offer Type"].classes_)

        submitted = st.form_submit_button("SCORE LEAD")

    if submitted:
        data = {
            "Customer Lifetime Value": clv, "Monthly Premium Auto": premium,
            "Sales Channel": channel, "Renew Offer Type": offer,
            "Education": "Bachelor", "EmploymentStatus": "Employed",
            "Gender": "M", "Location Code": "Suburban",
            "Coverage": "Basic", "Vehicle Class": "Two-Door Car",
            "Months Since Policy Inception": 12
        }
        X = pd.DataFrame([data])
        for col in X.select_dtypes("object").columns:
            if col in encoders:
                X[col] = encoders[col].transform(X[[col]])
        X = X[features]
        prob = model.predict_proba(X)[0, 1]
        st.success(f"**Propensity: {prob:.1%}**")

elif page == "Export Report":
    st.title("Export Marketing Report")
    if st.button("Generate PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Data-Driven Marketing KPIs", ln=1, align="C")
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Turning Insurance Leads into Lifetime Profit", ln=1, align="C")
        pdf.set_font("Arial", "I", 12)
        pdf.cell(0, 10, "By Howard Nguyen, PhD", ln=1, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=11)
        article = """
In an era where every ad dollar is scrutinized, four KPIs—Conversion Rate, Customer
Lifetime Value (CLV), Cost Per Acquisition (CPA), and Return on Investment (ROI)—are
the North Star for profitable growth...

This analysis of 9,000+ insurance prospects reveals a 14.3% overall conversion rate,
yet Agent-led sales hit 19.2% while Call Centers languish at 10.9%.

Data-science boost: Route high-propensity leads to agents → 34% lift.
        """
        pdf.multi_cell(0, 6, article)
        pdf.output("report.pdf")
        with open("report.pdf", "rb") as f:
            st.download_button("Download PDF", f, "Insurance_KPI_Report.pdf")