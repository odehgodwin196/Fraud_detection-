import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Fraud Detection System")

# Debug check
st.write("Files in directory:", os.listdir())

# Load model safely
try:
    model = joblib.load("fraud_model.pkl")
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

st.write("Enter transaction details:")

# Inputs
amount = st.number_input("Transaction Amount", min_value=0.0)
v14 = st.number_input("V14")
v7 = st.number_input("V7")

threshold = 0.6

if st.button("Predict"):

    input_data = np.zeros((1, 30))

    input_data[0, 0] = amount
    input_data[0, 14] = v14
    input_data[0, 7] = v7

    try:
        prob = model.predict_proba(input_data)[0][1]

        st.write(f"Fraud Probability: {prob:.2f}")

        if prob > threshold:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Legitimate Transaction")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
