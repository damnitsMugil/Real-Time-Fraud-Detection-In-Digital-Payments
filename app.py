import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# ==============================
# Load trained models
# ==============================
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
meta_model = joblib.load('meta_model.pkl')
dnn_model = tf.keras.models.load_model('dnn_model.h5')

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="centered")

st.title("💳 Real-Time Fraud Detection System")
st.write("Enter transaction details below to check if it’s **legit** or **fraudulent**.")

# Example: replace these with your dataset’s real feature names
feature_names = [
    "Transaction Amount", "Transaction Time", "Num of Past Transactions",
    "User Age", "Account Balance", "Device Trust Score"
]

# Input fields
inputs = []
for f in feature_names:
    val = st.number_input(f"Enter {f}:", value=0.0)
    inputs.append(val)

if st.button("🔍 Check Transaction"):
    X_input = np.array([inputs])
    
    # Scale/normalize if needed (assuming you used StandardScaler earlier)
    # from joblib import load
    # scaler = load('scaler.pkl')
    # X_input = scaler.transform(X_input)
    
    # Predictions from each model
    rf_pred = rf_model.predict_proba(X_input)[:, 1]
    dnn_pred = dnn_model.predict(X_input).flatten()
    xgb_pred = xgb_model.predict_proba(X_input)[:, 1]
    
    # Stack them for meta-model
    stacked_pred = np.column_stack((rf_pred, dnn_pred, xgb_pred))
    final_pred = meta_model.predict(stacked_pred)
    final_prob = meta_model.predict_proba(stacked_pred)[:, 1]

    # Output results
    st.subheader("🔎 Prediction Result:")
    if final_pred[0] == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Confidence: {final_prob[0]*100:.2f}%)")
    else:
        st.success(f"✅ Legit Transaction (Confidence: {(1 - final_prob[0])*100:.2f}%)")
