import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Fraud Detection System")
st.write("Enter transaction details:")

# Example inputs
amount = st.number_input("Transaction Amount", min_value=0.0)
v14 = st.number_input("V14")
v7 = st.number_input("V7")

threshold = 0.6

if st.button("Predict"):
    # Create a 30-feature vector (fill with zeros)
    input_data = np.zeros((1, 30))
    
    # Place values in the correct indices (adjust if needed)
    input_data[0, 0] = amount   # assuming 'Amount' was feature 0
    input_data[0, 14] = v14     # assuming V14 was feature 14
    input_data[0, 7] = v7       # assuming V7 was feature 7
    
    prob = model.predict_proba(input_data)[0][1]
    
    st.write(f"Fraud Probability: {prob:.2f}")
    
    if prob > threshold:
        st.error("🚨 Fraud Detected")
    else:
        st.success("✅ Legitimate Transaction")
