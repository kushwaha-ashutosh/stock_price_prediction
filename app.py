from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

app = Flask(__name__)

# Load trained model and scaler
model = load_model("model/stock_model.h5")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.form["ticker"].upper()

    try:
        # Download last 100 days of data
        stock = yf.Ticker(ticker)
        data = stock.history(period="100d")
        info = stock.info

        if data.empty:
            raise ValueError("No recent data found.")

        # Prepare input for prediction
        close_prices = data["Close"].values.reshape(-1, 1)
        scaled_data = scaler.transform(close_prices)
        last_60 = scaled_data[-60:]
        X_input = np.array(last_60).reshape(1, 60, 1)
        prediction = model.predict(X_input)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        # Recent stats
        current_price = close_prices[-1][0]
        change = current_price - close_prices[-2][0]
        pct_change = (change / close_prices[-2][0]) * 100
        volume = int(data["Volume"].iloc[-1])
        latest_date = data.index[-1].strftime("%Y-%m-%d")

        company_name = info.get("longName", ticker)
        logo_url = info.get("logo_url", "")
        sector = info.get("sector", "N/A")
        market_cap = info.get("marketCap", 0)
        market_cap_b = f"${market_cap/1e9:.2f}B" if market_cap else "N/A"

        return render_template(
            "index.html",
            prediction_text=f"📈 Predicted next closing price for {ticker} after {latest_date}: ${predicted_price:.2f}",
            company_name=company_name,
            logo_url=logo_url,
            ticker=ticker,
            current_price=current_price,
            pct_change=pct_change,
            volume=volume,
            sector=sector,
            market_cap=market_cap_b,
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error fetching data for {ticker}: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
