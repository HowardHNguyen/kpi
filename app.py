# app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from fpdf2 import FPDF
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Insurance KPI Dashboard", layout="wide")

# Load
model = joblib.load("best_model.pkl")
encoders = joblib.load("label_encoders.pkl")
features = joblib.load("feature_names.pkl")

with st.sidebar:
    page = option_menu("KPI Dashboard", [
        "Conversion Rate", "CLV Forecast", "CPA vs ROI",
        "Propensity Scorer", "Export Report"
    ])

# PAGE 1
if page == "Conversion Rate":
    st.title("Conversion Rate by Channel")
    df = pd.DataFrame({"Channel": ["Agent", "Web", "Branch", "Call Center"], "Rate": [19.2, 11.8, 11.9, 10.9]})
    st.bar_chart(df.set_index("Channel"))
    st.markdown("**Data-science boost**: Agent = **19.2%** (+34% vs 14.3% average)")

# PAGE 4
elif page == "Propensity Scorer":
    st.title("Real-Time Propensity Scorer")
    HIGH_IMPACT = { ... }  # [same as before]
    # [Full scorer code with SHAP]

# PAGE 5 – YOUR FULL ARTICLE
elif page == "Export Report":
    st.title("Export Marketing Report")
    if st.button("Generate PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Data-Driven Marketing KPIs", ln=1, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=11)
        article_text = """
        In an era where every ad dollar is scrutinized, four KPIs—Conversion Rate, Customer
        Lifetime Value (CLV), Cost Per Acquisition (CPA), and Return on Investment (ROI)—are
        the North Star for profitable growth. When powered by data science, these metrics
        evolve from static reports into predictive engines that allocate budget with surgical
        precision.

        1. Conversion Rate – The Pulse of Campaign Effectiveness
        Conversion Rate is the percentage of prospects who complete a desired action...
        This analysis of 9,000+ insurance prospects reveals a 14.3% overall conversion rate,
        yet Agent-led sales hit 19.2% while Call Centers languish at 10.9%.
        """
        pdf.multi_cell(0, 8, article_text)
        pdf.output("report.pdf")
        with open("report.pdf", "rb") as f:
            st.download_button("Download PDF", f, "Insurance_KPI_Report.pdf")