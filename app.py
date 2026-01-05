import streamlit as st
import pandas as pd
import joblib

import requests
from io import BytesIO

url = "https://huggingface.co/your-username/telco-churn-rf/resolve/main/churn_model.pkl"
model = joblib.load(BytesIO(requests.get(url).content))


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef2f3, #dfe9f3);
    }
    .main-title {
        text-align: center;
        color: #1f2c56;
        font-size: 42px;
        font-weight: 800;
    }
    .subtitle {
        text-align: center;
        color: #4b5563;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }
    .section-title {
        color: #111827;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/churn_model.pkl")

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict customer churn using a machine learning model</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧾 Customer Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "📄 Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )
    tenure = st.number_input(
        "⏳ Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )
    internetservice = st.selectbox(
        "🌐 Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:
    monthlycharges = st.number_input(
        "💰 Monthly Charges ($)",
        min_value=20.0,
        max_value=150.0,
        value=70.0,
        step=1.0
    )
    paymentmethod = st.selectbox(
        "💳 Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREPARE INPUT DATA ----------------
input_df = pd.DataFrame([{
    "contract": contract,
    "tenure": tenure,
    "monthlycharges": monthlycharges,
    "internetservice": internetservice,
    "paymentmethod": paymentmethod,

    # Defaults (model-required)
    "gender": "Female",
    "seniorcitizen": 0,
    "partner": "No",
    "dependents": "No",
    "phoneservice": "Yes",
    "multiplelines": "No",
    "onlinesecurity": "No",
    "onlinebackup": "No",
    "deviceprotection": "No",
    "techsupport": "No",
    "streamingtv": "No",
    "streamingmovies": "No",
    "paperlessbilling": "Yes",
    "totalcharges": monthlycharges * max(tenure, 1),
}])

# ---------------- PROFILE CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Customer Profile</div>', unsafe_allow_html=True)
st.dataframe(input_df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

if st.button("🚀 Predict Churn", use_container_width=True):
    churn_prob = model.predict_proba(input_df)[0][1]

    st.markdown('<div class="section-title">📈 Prediction Result</div>', unsafe_allow_html=True)
    st.metric("Churn Probability", f"{churn_prob:.2%}")

    if churn_prob >= 0.5:
        st.error("⚠️ High Risk of Churn")
        st.write(
            "This customer has a **high probability of churning**. "
            "Retention strategies such as better contracts or discounts are recommended."
        )
    else:
        st.success("✅ Low Risk of Churn")
        st.write("This customer appears stable with a low likelihood of churn.")

st.markdown("</div>", unsafe_allow_html=True)



