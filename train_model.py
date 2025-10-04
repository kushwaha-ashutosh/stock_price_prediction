# train_model.py
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ===============================
# CONFIG
# ===============================
ticker = "AAPL"  # you can change this to any stock like 'GOOG', 'TSLA', etc.
start_date = "2015-01-01"
end_date = "2025-10-01"

# Create data folder
os.makedirs("data", exist_ok=True)

# ===============================
# STEP 1: DOWNLOAD & CLEAN DATA
# ===============================
print(f"📥 Downloading data for {ticker} ...")
df = yf.download(ticker, start=start_date, end=end_date).reset_index()
df = df[['Date', 'Close']]
df.to_csv(f"data/{ticker}.csv", index=False)
print(f"✅ Data downloaded and saved to data/{ticker}.csv")

# ===============================
# STEP 2: FEATURE SCALING
# ===============================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(df['Close']).reshape(-1, 1))

# ===============================
# STEP 3: TRAIN-TEST SPLIT
# ===============================
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# ===============================
# STEP 4: CREATE SEQUENCES
# ===============================
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ===============================
# STEP 5: BUILD MODEL
# ===============================
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# ===============================
# STEP 6: TRAIN MODEL
# ===============================
print("🚀 Training model (this may take a few minutes)...")
history = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

# ===============================
# STEP 7: PREDICTIONS
# ===============================
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# ===============================
# STEP 8: EVALUATION
# ===============================
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(actual, predictions))
print(f"📊 RMSE: {rmse:.2f}")

# ===============================
# STEP 9: VISUALIZATION
# ===============================
plt.figure(figsize=(10, 6))
plt.plot(df['Date'][-len(actual):], actual, color='blue', label='Actual Price')
plt.plot(df['Date'][-len(predictions):], predictions, color='red', label='Predicted Price')
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# STEP 10: SAVE MODEL
# ===============================
model.save("model/stock_model.h5")
import joblib
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")
print("✅ Model and scaler saved in 'model/' folder")
