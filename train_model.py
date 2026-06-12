import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ----------------------------------
# LOAD DATASET
# ----------------------------------

df = pd.read_csv("data/AirPassengers.csv")

# Passenger column
data = df["#Passengers"].values.reshape(-1, 1)

# ----------------------------------
# SCALING
# ----------------------------------

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data)

# ----------------------------------
# CREATE SEQUENCES
# ----------------------------------

X = []
y = []

time_step = 12

for i in range(time_step, len(data_scaled)):
    X.append(data_scaled[i - time_step : i, 0])

    y.append(data_scaled[i, 0])

X = np.array(X)
y = np.array(y)

# reshape for LSTM

X = X.reshape(X.shape[0], X.shape[1], 1)

print("X Shape:", X.shape)
print("y Shape:", y.shape)

# ----------------------------------
# BUILD MODEL
# ----------------------------------

model = Sequential(
    [
        LSTM(units=50, return_sequences=False, input_shape=(time_step, 1)),
        Dropout(0.2),
        Dense(1),
    ]
)

# ----------------------------------
# COMPILE
# ----------------------------------

model.compile(optimizer="adam", loss="mean_squared_error")

# ----------------------------------
# TRAIN
# ----------------------------------

history = model.fit(X, y, epochs=20, batch_size=16, verbose=1)

# ----------------------------------
# SAVE MODEL
# ----------------------------------

model.save("models/lstm_model.keras")

# ----------------------------------
# SAVE SCALER
# ----------------------------------

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model Saved Successfully")
print("Scaler Saved Successfully")
