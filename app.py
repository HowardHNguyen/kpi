# app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance Propensity Scorer", layout="wide")
st.title("Propensity-to-Buy Scorer")
st.markdown(
    "### Real-time* conversion probability + SHAP explanation – by **Howard Nguyen, PhD**"
)
st.caption(
    "From the article: *[Data-Driven Marketing KPIs: Turning Insurance Leads into Lifetime Profit]*"
)

# ----------------------------------------------------------------------
# Load model, encoders and feature list
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("propensity_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    features = joblib.load("feature_names.pkl")
    return model, encoders, features


model, encoders, features = load_artifacts()

# ----------------------------------------------------------------------
# UI – form
# ----------------------------------------------------------------------
with st.form("lead_form"):
    col1, col2 = st.columns(2)
    with col1:
        clv = st.number_input(
            "Customer Lifetime Value ($)", min_value=0.0, value=8000.0, step=100.0
        )
        premium = st.number_input(
            "Monthly Premium Auto ($)", min_value=0.0, value=100.0, step=10.0
        )
        months = st.number_input(
            "Months Since Policy Inception", min_value=0, value=12, step=1
        )
    with col2:
        channel = st.selectbox(
            "Sales Channel", options=encoders["Sales Channel"].classes_
        )
        offer = st.selectbox(
            "Renew Offer Type", options=encoders["Renew Offer Type"].classes_
        )
        education = st.selectbox(
            "Education", options=encoders["Education"].classes_
        )

    col3, col4 = st.columns(2)
    with col3:
        employment = st.selectbox(
            "Employment Status", options=encoders["EmploymentStatus"].classes_
        )
        gender = st.selectbox("Gender", options=encoders["Gender"].classes_)
        location = st.selectbox(
            "Location Code", options=encoders["Location Code"].classes_
        )
    with col4:
        coverage = st.selectbox("Coverage", ["Basic", "Extended", "Premium"])
        vehicle = st.text_input("Vehicle Class", "Two-Door Car")

    submitted = st.form_submit_button("SCORE LEAD")

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if submitted:
    with st.spinner("Scoring…"):
        # ---- 1. numeric / engineered features -------------------------
        data = {
            "Customer Lifetime Value": clv,
            "Monthly Premium Auto": premium,
            "Months_Since_Start": months,
            "Premium_Policy": 1 if coverage == "Premium" else 0,
            "Luxury_Vehicle": 1 if "Luxury" in vehicle else 0,
        }

        # ---- 2. categorical encoding (ONE LOOP ONLY) -----------------
        cat_mapping = {
            "Sales Channel": channel,
            "Renew Offer Type": offer,
            "Education": education,
            "EmploymentStatus": employment,
            "Gender": gender,
            "Location Code": location,
        }

        encoded_debug = {}
        for col, val in cat_mapping.items():
            le = encoders[col]
            try:
                encoded_val = le.transform([val])[0]
            except ValueError:  # unseen category → fallback to first class
                encoded_val = 0
            data[col] = encoded_val
            encoded_debug[col] = (val, encoded_val)

        # ---- 3. final dataframe --------------------------------------
        X = pd.DataFrame([data])[features]
        prob = model.predict_proba(X)[0, 1]

        # ---- 4. results ----------------------------------------------
        st.success(f"**Propensity Score: {prob:.1%}**")
        st.metric(
            "Recommended Action",
            "Route to Agent" if prob > 0.7 else "Web Funnel",
        )

        # ---- 5. article-style message ---------------------------------
        st.markdown("---")
        if prob > 0.7:
            msg = f"""
            <div style='background-color:#e6f7e6;padding:15px;border-radius:10px;font-size:15px;line-height:1.6;'>
            <strong>High-value lead: {prob:.1%} conversion probability</strong><br>
            <strong>Recommended Action:</strong> <strong>Route to Agent</strong> — maximize lifetime value.
            </div>
            """
        else:
            msg = f"""
            <div style='background-color:#f0f2f6;padding:15px;border-radius:10px;font-size:15px;line-height:1.6;'>
            <strong>Only {prob:.1%} chance this lead will convert</strong> — too low for agent routing.<br>
            <strong>Recommended Action:</strong> Send to <strong>Web Funnel</strong> (automated emails, retargeting).
            </div>
            """
        st.markdown(msg, unsafe_allow_html=True)
        st.markdown("---")

        # ---- 6. optional debug expander -------------------------------
        with st.expander("Debug – encoded values (remove in production)", expanded=False):
            st.json(encoded_debug)

        # ---- 7. SHAP waterfall ----------------------------------------
        with st.spinner("Generating SHAP explanation…"):
            try:
                # model-agnostic explainer works with any tree model
                explainer = shap.Explainer(
                    lambda x: model.predict_proba(x)[:, 1], X, feature_names=features
                )
                shap_values = explainer(X)

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_values[0], max_display=12, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.info(f"SHAP plot skipped: {e}")