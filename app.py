import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("models/best_churn_model.pkl")
    with open("models/model_metadata.json") as f:
        metadata = json.load(f)
    return pipeline, metadata

pipeline, metadata = load_artifacts()
threshold = metadata["selected_threshold"]
st.title("Customer Churn Predictor")
st.caption(
    f"Model: {metadata['model_name']}  |  "
    f"Test ROC-AUC: {metadata['test_metrics']['roc_auc']:.3f}  |  "
    f"Decision threshold: {threshold}"
)
st.sidebar.header("Customer profile")
st.sidebar.write(
    "Fill in the customer's details. These are the exact raw fields the model "
    "was trained on (no customer ID is needed - it carries no predictive signal)."
)
with st.sidebar:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )
    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )
    partner = st.selectbox(
        "Has Partner",
        ["No", "Yes"]
    )
    dependents = st.selectbox(
        "Has Dependents",
        ["No", "Yes"]
    )
    tenure = st.slider(
        "Tenure (months)",
        0,
        72,
        12
    )
    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=65.0,
        step=0.5
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=round(monthly_charges * tenure, 2),
        step=1.0
    )


# ---------------------------------------------------------------------------
# Build raw input row
# Saved pipeline handles preprocessing internally
# ---------------------------------------------------------------------------

input_row = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}])

if st.button("Predict Churn", type="primary"):
    churn_proba = pipeline.predict_proba(input_row)[0, 1]
    stay_proba = 1 - churn_proba
    is_high_risk = churn_proba >= threshold
    col1, col2 = st.columns(2)
    col1.metric(
        "Churn Probability",
        f"{churn_proba * 100:.1f}%"
    )
    col2.metric(
        "Stay Probability",
        f"{stay_proba * 100:.1f}%"
    )
    st.progress(
        min(max(churn_proba, 0.0), 1.0)
    )
    if is_high_risk:
        st.error("⚠️ High Churn Risk")
        st.markdown(
            """
            This customer has a high predicted churn risk.

            Consider proactive retention actions such as:

            - personalized offers
            - support assistance
            - contract incentives
            - targeted retention campaigns
            """
        )
    else:
        st.success("✅ Low Churn Risk")
        st.markdown(
            "This customer currently has a lower predicted churn risk."
        )
st.caption(
    "This tool loads a pre-trained pipeline; no model training happens in this app. "
    f"Threshold ({threshold}) was selected via out-of-fold F1 optimization on "
    "training data, not on the held-out test set."
)