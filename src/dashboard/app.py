# src/dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import json

# Configuration - change this when deployed!
API_URL = "https://enterprise-customer-churn-prediction.onrender.com/predict"  # ← Use this when deployed

st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Telco Customer Churn Prediction Dashboard")
st.markdown("""
Fill in the customer details below to get a **real-time churn prediction** from the XGBoost model via FastAPI backend.
""")

# ────────────────────────────────────────────────
# Input Form
# ────────────────────────────────────────────────
with st.form("customer_form", clear_on_submit=False):
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=80, value=12, step=1)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_bk = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        stream_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    col3, col4 = st.columns(2)

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])

    with col4:
        monthly_ch = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.35, step=0.01)
        total_ch = st.number_input("Total Charges ($)", min_value=0.0, value=843.25, step=0.01)

    submitted = st.form_submit_button("Get Churn Prediction", type="primary", use_container_width=True)

# ────────────────────────────────────────────────
# Prediction Logic
# ────────────────────────────────────────────────
if submitted:
    # Build payload with correct types
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bk,
        "DeviceProtection": device_prot,
        "TechSupport": tech_sup,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly_ch),
        "TotalCharges": float(total_ch)
    }

    # Show what is being sent (very useful for debugging)
    st.subheader("Payload being sent to API")
    st.json(payload)

    with st.spinner("Getting prediction from backend..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=12)
            response.raise_for_status()  # Raises exception for 4xx/5xx

            result = response.json()

            st.success("Prediction received successfully!")

            # Display results in nice metrics
            col_a, col_b = st.columns(2)
            prob = result.get('churn_probability', 0)
            pred = result.get('predicted_churn', 'Unknown')

            col_a.metric(
                label="Churn Probability",
                value=f"{prob * 100:.1f}%",
                delta_color="inverse" if pred == "Yes" else "normal"
            )
            col_b.metric(
                label="Predicted to Churn?",
                value=pred,
                delta_color="inverse" if pred == "Yes" else "normal"
            )

            # Optional: confidence if available
            if 'confidence' in result:
                st.metric("Model Confidence", f"{result['confidence']*100:.1f}%")

            # Show full response
            with st.expander("Full API Response (for debugging)"):
                st.json(result)

        except requests.exceptions.HTTPError as http_err:
            st.error(f"Backend returned an error (HTTP {response.status_code})")
            try:
                error_detail = response.json()
                st.warning("Error details from backend:")
                st.json(error_detail)
            except:
                st.write("Raw response:", response.text)

        except requests.exceptions.ConnectionError:
            st.error("Connection failed. Is the FastAPI backend running on http://127.0.0.1:8000 ?")
            st.info("Start backend with: `uvicorn src.serving.app:app --reload`")

        except requests.exceptions.Timeout:
            st.error("Request timed out. Backend may be slow or not responding.")

        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Backend: FastAPI | Model: XGBoost | Dashboard: Streamlit | Created by Ganesh")