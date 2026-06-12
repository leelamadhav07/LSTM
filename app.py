import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# ----------------------------------
# LOAD MODEL
# ----------------------------------

model = tf.keras.models.load_model("models/lstm_model.keras")

# ----------------------------------
# LOAD SCALER
# ----------------------------------

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------------
# PAGE
# ----------------------------------

st.set_page_config(page_title="LSTM Forecasting", layout="centered")

st.title("Air Passenger Forecasting using LSTM")

st.write("Enter Passenger Counts for Previous 12 Months")

# ----------------------------------
# INPUTS
# ----------------------------------

values = []

for i in range(12):
    value = st.number_input(f"Month {i + 1}", min_value=0, value=100)

    values.append(value)

# ----------------------------------
# PREDICTION
# ----------------------------------

if st.button("Predict Next Month"):
    data = np.array(values).reshape(-1, 1)

    # scale

    data_scaled = scaler.transform(data)

    # reshape

    data_scaled = data_scaled.reshape(1, 12, 1)

    prediction = model.predict(data_scaled)

    prediction = scaler.inverse_transform(prediction)

    st.success(f"Predicted Passengers : {prediction[0][0]:.0f}")
