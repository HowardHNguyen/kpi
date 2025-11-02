# app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from fpdf2 import FPDF
import base64
from streamlit_option_menu import option_menu
import os

# ----------------------------------------------------------------------
# Load artifacts (they are in the same folder as app.py)
# ----------------------------------------------------------------------
model = joblib.load("best_model.pkl")
encoders = joblib.load("label_encoders.pkl")
features = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Insurance KPI Dashboard", layout="wide")

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
with st.sidebar:
    page = option_menu(
        "KPI Dashboard",
        ["Conversion Rate", "CLV Forecast", "CPA vs ROI",
         "Propensity Scorer", "Export Report"],
        icons=['bar-chart', 'graph-up', 'currency-dollar', 'robot', 'file-earmark-pdf']
    )

# ----------------------------------------------------------------------
# PAGE 1 – Conversion Rate by Channel (exact numbers from your article)
# ----------------------------------------------------------------------
if page == "Conversion Rate":
    st.title("Conversion Rate by Channel")
    data = {
        "Channel": ["Agent", "Web", "Branch", "Call Center"],
        "Rate": [19.2, 11.8, 11.9, 10.9]
    }
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index("Channel"))
    st.markdown("""
    **Data-science boost** – Agent-led routing = **19.2 %**  
    (vs. overall average **14.3 %** → **34 % lift**)
    """)

# ----------------------------------------------------------------------
# PAGE 2 – CLV Forecast (simple linear proxy – you can replace later)
# ----------------------------------------------------------------------
elif page == "CLV Forecast":
    st.title("Customer Lifetime Value Forecast")
    clv = st.number_input("Customer Lifetime Value ($)", min_value=0.0, value=8000.0, step=100.0)
    premium = st.number_input("Monthly Premium Auto ($)", min_value=0.0, value=100.0, step=10.0)
    months = st.number_input("Months Since Policy Inception", min_value=0, value=12, step=1)
    if st.button("Forecast CLV"):
        # Very simple model – replace with a real regressor later
        forecast = clv + premium * months * 0.8
        st.success(f"**Forecasted CLV: ${forecast:,.0f}**")

# ----------------------------------------------------------------------
# PAGE 3 – CPA vs ROI Heatmap (static from article)
# ----------------------------------------------------------------------
elif page == "CPA vs ROI":
    st.title("CPA vs ROI Heatmap")
    st.image("https://i.imgur.com/5vXjK9P.png", caption="Agent = Green (high ROI), Call Center = Red (low ROI)")

# ----------------------------------------------------------------------
# PAGE 4 – Real-Time Propensity Scorer (your 92 % golden-lead scorer)
# ----------------------------------------------------------------------
elif page == "Propensity Scorer":
    st.title("Real-Time Propensity-to-Buy Scorer")

    # ----- High-impact auto-fill -----
    HIGH_IMPACT = {
        "Customer Lifetime Value": 15000,
        "Monthly Premium Auto": 200,
        "Months Since Policy Inception": 6,
        "Sales Channel": "Agent",
        "Renew Offer Type": "Offer2",
        "Education": "Doctor",
        "EmploymentStatus": "Retired",
        "Gender": "F",
        "Location Code": "Suburban",
        "Coverage": "Premium",
        "Vehicle Class": "Luxury SUV"
    }

    with st.form("lead_form"):
        col1, col2 = st.columns(2)
        with col1:
            clv = st.number_input("Customer Lifetime Value ($)", min_value=0.0, value=8000.0, step=100.0)
            premium = st.number_input("Monthly Premium Auto ($)", min_value=0.0, value=100.0, step=10.0)
            months = st.number_input("Months Since Policy Inception", min_value=0, value=12, step=1)
        with col2:
            channel = st.selectbox("Sales Channel", options=encoders["Sales Channel"].classes_)
            offer = st.selectbox("Renew Offer Type", options=encoders["Renew Offer Type"].classes_)
            education = st.selectbox("Education", options=encoders["Education"].classes_)

        col3, col4 = st.columns(2)
        with col3:
            employment = st.selectbox("Employment Status", options=encoders["EmploymentStatus"].classes_)
            gender = st.selectbox("Gender", options=encoders["Gender"].classes_)
            location = st.selectbox("Location Code", options=encoders["Location Code"].classes_)
        with col4:
            coverage = st.selectbox("Coverage", ["Basic", "Extended", "Premium"])
            vehicle_options = ["Luxury SUV", "Luxury Car", "Sports Car", "SUV", "Two-Door Car", "Four-Door Car"]
            vehicle = st.selectbox("Vehicle Class", options=vehicle_options)

        submitted = st.form_submit_button("SCORE LEAD")

    # ----- Auto-fill 90 %+ button -----
    if st.button("Load 90 %+ High-Conversion Lead", type="primary"):
        for k, v in HIGH_IMPACT.items():
            st.session_state[k] = v
        st.rerun()

    # Load from session state if present
    for k, v in HIGH_IMPACT.items():
        if k in st.session_state:
            globals()[k.lower().replace(" ", "_")] = st.session_state[k]

    if submitted:
        # ----- Build feature vector -----
        data = {
            "Customer Lifetime Value": clv,
            "Monthly Premium Auto": premium,
            "Months Since Policy Inception": months,
            "Premium_Policy": 1 if coverage == "Premium" else 0,
            "Luxury_Vehicle": 1 if "Luxury" in vehicle else 0,
        }

        cat_map = {
            "Sales Channel": channel,
            "Renew Offer Type": offer,
            "Education": education,
            "EmploymentStatus": employment,
            "Gender": gender,
            "Location Code": location,
        }

        for col, val in cat_map.items():
            le = encoders[col]
            try:
                data[col] = le.transform([val])[0]
            except ValueError:
                data[col] = 0

        X = pd.Datazmann([data])[features]
        prob = model.predict_proba(X)[0, 1]

        # ----- Results -----
        st.success(f"**Propensity Score: {prob:.1%}**")
        st.metric("Recommended Action", "Route to Agent" if prob > 0.70 else "Web Funnel")

        # ----- SHAP waterfall -----
        with st.spinner("Generating SHAP explanation…"):
            explainer = shap.Explainer(lambda x: model.predict_proba(x)[:, 1], X, feature_names=features)
            shap_vals = explainer(X)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_vals[0], max_display=12, show=False)
            st.pyplot(fig)

# ----------------------------------------------------------------------
# PAGE 5 – Export PDF (your full article embedded)
# ----------------------------------------------------------------------
elif page == "Export Report":
    st.title("Export Marketing Report")
    if st.button("Generate PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Data-Driven Marketing KPIs", ln=1, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        txt = """
        In an era where every ad dollar is scrutinized, four KPIs—Conversion Rate,
        Customer Lifetime Value (CLV), Cost Per Acquisition (CPA), and Return on
        Investment (ROI)—are the North Star for profitable growth. When powered by
        data science, these metrics evolve from static reports into predictive
        engines that allocate budget with surgical precision.

        1. Conversion Rate – The Pulse of Campaign Effectiveness
        ...
        This analysis of 9,000+ insurance prospects reveals a 14.3 % overall
        conversion rate, yet Agent-led sales hit 19.2 % while Call Centers
        languish at 10.9 %.
        """
        pdf.multi_cell(0, 10, txt)
        pdf.output("kpi_report.pdf")

        with open("kpi_report.pdf", "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="Insurance_KPI_Report.pdf",
                mime="application/pdf"
            )