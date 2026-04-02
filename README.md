# 📈 Stock Price Prediction — LSTM Deep Learning + Streamlit Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![LSTM](https://img.shields.io/badge/LSTM-Time%20Series-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A complete stock price prediction system using Stacked LSTM neural networks trained on live market data. Includes a professional Streamlit dashboard where users can type any stock symbol and get predictions instantly.

---

## 🎯 Features

- ✅ Stacked LSTM (3 layers) for time series prediction
- ✅ Live data via Yahoo Finance API — works for ANY stock
- ✅ 30-day future price forecasting
- ✅ Moving averages (MA20, MA50, MA200)
- ✅ Multi-stock comparison chart
- ✅ Interactive Streamlit dashboard with dark theme
- ✅ predict_stock() — predict any stock in one function
- ✅ Day-by-day forecast table

---

## 📊 Results on Real Market Data

| Stock | R² Score | MAE | 30-Day Forecast |
|---|---|---|---|
| GOOGL (Google) | **0.8324** | $6.16 | $161.20 (-15.32%) |
| TSLA (Tesla) | **0.9131** | $12.19 | $258.94 (-37.97%) |

> ⚠️ **Disclaimer:** This is an educational project. Not financial advice. Stock markets are inherently unpredictable.

---

## 🛠️ Tech Stack

- Python 3.9+
- TensorFlow 2.x / Keras
- yfinance (live data)
- Scikit-learn
- Matplotlib
- Streamlit
- NumPy
- Pandas

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/engineerzainabbas/stock-price-lstm.git
cd stock-price-lstm
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook stock_price_lstm.ipynb
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run stock_dashboard.py
```

### 5. Predict any stock
```python
predict_stock('AAPL')   # Apple
predict_stock('GOOGL')  # Google
predict_stock('TSLA')   # Tesla
predict_stock('MSFT')   # Microsoft
predict_stock('AMZN')   # Amazon
```

---

## 📂 Project Structure

```
stock-price-lstm/
│
├── stock_price_lstm.ipynb    ← Main notebook
├── stock_dashboard.py        ← Streamlit dashboard
├── requirements.txt           ← Dependencies
├── README.md                  ← This file
└── results/                   ← Output charts
    ├── stocks_comparison.png
    ├── moving_averages.png
    ├── training_history.png
    ├── actual_vs_predicted.png
    └── future_forecast.png
```

---

## 🧠 Model Architecture

```
Input (60 days × 1 feature)
→ LSTM(128, return_sequences=True) + Dropout(0.2)
→ LSTM(64, return_sequences=True)  + Dropout(0.2)
→ LSTM(32, return_sequences=False) + Dropout(0.2)
→ Dense(16, relu)
→ Dense(1)
```

| Parameter | Value |
|---|---|
| Sequence Length | 60 days |
| Training Data | 2018–2024 (6 years) |
| Optimizer | Adam (lr=0.001) |
| Loss | Mean Squared Error |
| Early Stopping | patience=8 |

---

## 🖥️ Dashboard Features

| Feature | Description |
|---|---|
| Any stock symbol | Type AAPL, TSLA, GOOGL, etc. |
| Price metrics | Current price, daily change, 52W high/low |
| Moving averages | MA20, MA50, MA200 chart |
| Stock comparison | Normalize and compare multiple stocks |
| LSTM training | Live progress bar while training |
| Future forecast | 30-day day-by-day prediction table |
| Dark theme | Professional dark UI |

---

## 🏭 Real-World Applications

- 💰 **Finance** — Investment research tools
- 📊 **Trading** — Algorithmic trading signals
- 🏦 **Banking** — Portfolio risk management
- 📱 **FinTech** — Smart investment apps
- 🎓 **Education** — Time series learning

---

## 💼 Need a Custom Solution?

If you need a custom stock prediction or time series forecasting system:

- 🎯 Custom LSTM/GRU model for your financial data
- 📊 Interactive dashboard for your portfolio
- 🔧 API for real-time predictions
- 📈 Backtesting framework

📧 engineerzainabbas@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/engineerzainabbas/)

---

## 👤 Author

**Zain Abbas**
- 🎓 Computer Engineering
- 💼 AI & Machine Learning Engineer
- 🔗 [LinkedIn](https://www.linkedin.com/in/engineerzainabbas/)
- 📧 engineerzainabbas@gmail.com

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock price predictions are inherently uncertain. Do not use this for real financial decisions.

---

## 📄 License

This project is licensed under the MIT License.
