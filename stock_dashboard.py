"""
📈 Stock Price Prediction Dashboard
Built with Streamlit + LSTM
Author: Zain Abbas
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from keras.layers import LSTM, Dropout, Dense

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title='Stock Price Predictor',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a3a5c, #2c3e50);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2980b9;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #3498db;
    }
    .metric-label {
        font-size: 13px;
        color: #95a5a6;
        margin-top: 5px;
    }
    .positive { color: #2ecc71 !important; }
    .negative { color: #e74c3c !important; }
    .stButton > button {
        background: linear-gradient(135deg, #2980b9, #1a3a5c);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3498db, #2c3e50);
    }
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────
@st.cache_data(ttl=3600)
def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)

    # Flatten MultiIndex columns if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm(seq_len):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


def predict_future(model, data, scaler, seq_len, n_days):
    last_seq = scaler.transform(data[-seq_len:].reshape(-1, 1))
    preds = []
    curr_seq = last_seq.copy()

    for _ in range(n_days):
        x_in = curr_seq[-seq_len:].reshape(1, seq_len, 1)
        next_p = model.predict(x_in, verbose=0)[0, 0]
        preds.append(next_p)
        curr_seq = np.append(curr_seq, [[next_p]], axis=0)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Predictor")
    st.markdown("---")

    ticker = st.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter any stock symbol: AAPL, GOOGL, TSLA, MSFT, AMZN, etc."
    ).upper()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime("2018-01-01")
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=pd.to_datetime("2024-12-31")
        )

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")

    seq_len = st.slider(
        "Sequence Length (days)",
        30, 120, 60,
        help="Days of history to use for each prediction"
    )
    epochs = st.slider("Training Epochs", 10, 100, 50)
    future_days = st.slider("Forecast Days", 7, 90, 30)
    batch_size = st.select_slider("Batch Size", [8, 16, 32, 64], value=32)

    st.markdown("---")
    predict_btn = st.button("🚀 Run Prediction")

    st.markdown("---")
    st.markdown("### 📊 Compare Stocks")
    compare_stocks = st.multiselect(
        "Select stocks to compare",
        ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "META", "NVDA", "NFLX"],
        default=["AAPL", "TSLA"]
    )
    compare_btn = st.button("📊 Compare")


# ── MAIN CONTENT ──────────────────────────────────────────────
st.markdown("# 📈 Stock Price Prediction Dashboard")
st.markdown("### Real-time predictions using LSTM Deep Learning")
st.markdown("---")

# ── STOCK INFO & EDA (always shown) ──────────────────────────
if ticker:
    with st.spinner(f'Loading {ticker} data...'):
        df = download_data(ticker, str(start_date), str(end_date))

    if df.empty:
        st.error(f"Could not find data for {ticker}. Please check the symbol.")
        st.stop()

    # Ensure required columns exist
    required_cols = ['Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Required column '{col}' not found in downloaded data.")
            st.stop()

    # Clean important columns
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna(subset=['Close', 'Volume'])

    if len(df) < 250:
        st.warning("Not enough data available for reliable analysis and prediction.")
        st.stop()

    # ── Quick stats ───────────────────────────────────────────
    current = float(df['Close'].iloc[-1])
    prev = float(df['Close'].iloc[-2])
    change = current - prev
    pct_chg = (change / prev) * 100
    high_52w = float(df['Close'].tail(252).max())
    low_52w = float(df['Close'].tail(252).min())
    avg_vol = int(df['Volume'].tail(30).mean())

    st.markdown(f"### {ticker} — Live Market Data")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>${current:.2f}</div>
            <div class='metric-label'>Current Price</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        color = 'positive' if change >= 0 else 'negative'
        sign = '+' if change >= 0 else ''
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value {color}'>{sign}{pct_chg:.2f}%</div>
            <div class='metric-label'>Daily Change</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#2ecc71'>${high_52w:.2f}</div>
            <div class='metric-label'>52W High</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#e74c3c'>${low_52w:.2f}</div>
            <div class='metric-label'>52W Low</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:#f39c12'>{avg_vol:,}</div>
            <div class='metric-label'>Avg Volume (30d)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price chart with moving averages ─────────────────────
    st.markdown("### 📊 Price History & Moving Averages")

    df_ma = df.copy()
    df_ma['Close'] = pd.to_numeric(df_ma['Close'], errors='coerce')
    df_ma['Volume'] = pd.to_numeric(df_ma['Volume'], errors='coerce')
    df_ma['Close'] = df_ma['Close'].ffill().bfill()
    df_ma['Volume'] = df_ma['Volume'].ffill().bfill()

    df_ma['MA20'] = df_ma['Close'].rolling(20).mean()
    df_ma['MA50'] = df_ma['Close'].rolling(50).mean()
    df_ma['MA200'] = df_ma['Close'].rolling(200).mean()

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8),
        facecolor='#0e1117',
        gridspec_kw={'height_ratios': [3, 1]}
    )

    for ax in axes:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#2c3e50')

    x = df_ma.index
    y_close = df_ma['Close'].to_numpy().flatten()
    y_ma20 = df_ma['MA20'].to_numpy().flatten()
    y_ma50 = df_ma['MA50'].to_numpy().flatten()
    y_ma200 = df_ma['MA200'].to_numpy().flatten()
    y_vol = df_ma['Volume'].to_numpy().flatten()

    axes[0].plot(x, y_close, color='#3498db', lw=1.5, label='Close')
    axes[0].plot(x, y_ma20, color='#e74c3c', lw=1, label='MA20')
    axes[0].plot(x, y_ma50, color='#2ecc71', lw=1, label='MA50')
    axes[0].plot(x, y_ma200, color='#f39c12', lw=1, label='MA200')
    axes[0].fill_between(x, y_close, alpha=0.05, color='#3498db')

    axes[0].set_title(f'{ticker} Price History', color='white', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Price (USD)', color='white')
    axes[0].legend(facecolor='#1a1a2e', labelcolor='white')
    axes[0].grid(alpha=0.15, color='gray')

    axes[1].bar(x, y_vol, color='#2ecc71', alpha=0.5, width=1)
    axes[1].set_title('Volume', color='white', fontsize=11)
    axes[1].set_ylabel('Volume', color='white')
    axes[1].grid(alpha=0.15, color='gray')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── STOCK COMPARISON ─────────────────────────────────────────
if compare_btn and compare_stocks:
    st.markdown("---")
    st.markdown("### 📊 Stock Performance Comparison")

    with st.spinner("Loading comparison data..."):
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='#0e1117')
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#2c3e50')

        colors_cmp = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12',
                      '#9b59b6', '#1abc9c', '#e67e22', '#fd79a8']

        for s, color in zip(compare_stocks, colors_cmp):
            d = download_data(s, str(start_date), str(end_date))
            if not d.empty and 'Close' in d.columns:
                d['Close'] = pd.to_numeric(d['Close'], errors='coerce')
                d = d.dropna(subset=['Close'])

                if len(d) > 0:
                    norm = (d['Close'] / d['Close'].iloc[0]) * 100
                    ax.plot(d.index, norm.to_numpy().flatten(), color=color, lw=2, label=s)

        ax.axhline(100, color='gray', linestyle='--', lw=1, alpha=0.5)
        ax.set_title('Normalized Performance (Base = 100)',
                     color='white', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalized Price', color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=11)
        ax.grid(alpha=0.15, color='gray')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── LSTM PREDICTION ──────────────────────────────────────────
if predict_btn:
    st.markdown("---")
    st.markdown(f"### 🧠 LSTM Prediction — {ticker}")

    if len(df) <= seq_len + 50:
        st.error("Not enough data for the selected sequence length. Try reducing sequence length or increasing date range.")
        st.stop()

    with st.spinner("Preparing data..."):
        close_vals = df[['Close']].values.astype(float)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_vals)

        X, y = create_sequences(scaled, seq_len)

        if len(X) < 50:
            st.error("Not enough sequence samples for training. Please use a larger date range.")
            st.stop()

        split = int(len(X) * 0.8)

        X_tr = X[:split].reshape(-1, seq_len, 1)
        X_te = X[split:].reshape(-1, seq_len, 1)
        y_tr, y_te = y[:split], y[split:]

    progress_bar = st.progress(0)
    status_text = st.empty()

    class StreamlitCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            prog = int((epoch + 1) / epochs * 100)
            progress_bar.progress(prog)
            status_text.text(
                f"Training epoch {epoch+1}/{epochs} | "
                f"Loss: {logs.get('loss', 0):.6f} | "
                f"Val Loss: {logs.get('val_loss', 0):.6f}"
            )

    with st.spinner("Training LSTM model..."):
        model = build_lstm(seq_len)
        model.fit(
            X_tr, y_tr,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                StreamlitCallback(),
                EarlyStopping(patience=8, restore_best_weights=True)
            ]
        )

    progress_bar.progress(100)
    status_text.text("Training complete!")

    # Predictions
    y_pred = scaler.inverse_transform(model.predict(X_te, verbose=0))
    y_act = scaler.inverse_transform(y_te.reshape(-1, 1))

    mae_v = mean_absolute_error(y_act, y_pred)
    rmse_v = np.sqrt(mean_squared_error(y_act, y_pred))
    r2_v = r2_score(y_act, y_pred)

    # Avoid divide by zero in MAPE
    non_zero_mask = y_act.flatten() != 0
    if np.any(non_zero_mask):
        mape_v = np.mean(np.abs((y_act.flatten()[non_zero_mask] - y_pred.flatten()[non_zero_mask]) / y_act.flatten()[non_zero_mask])) * 100
    else:
        mape_v = 0.0

    # Metrics
    st.markdown("#### 📊 Model Performance")
    m1, m2, m3, m4 = st.columns(4)

    metrics = [
        (m1, "MAE", f"${mae_v:.2f}"),
        (m2, "RMSE", f"${rmse_v:.2f}"),
        (m3, "R² Score", f"{r2_v:.4f}"),
        (m4, "MAPE", f"{mape_v:.2f}%"),
    ]

    for col, label, val in metrics:
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs predicted chart
    st.markdown("#### 📈 Actual vs Predicted")
    test_dates = df.index[seq_len + split:]

    fig2, ax2 = plt.subplots(figsize=(14, 6), facecolor='#0e1117')
    ax2.set_facecolor('#0e1117')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#2c3e50')

    ax2.plot(test_dates, y_act.flatten(), color='#3498db', lw=2, label='Actual')
    ax2.plot(test_dates, y_pred.flatten(), color='#e74c3c', lw=2,
             linestyle='--', label='Predicted')

    ax2.set_title(f'{ticker} Actual vs Predicted | R²={r2_v:.4f} | MAE=${mae_v:.2f}',
                  color='white', fontweight='bold')
    ax2.set_ylabel('Price (USD)', color='white')
    ax2.legend(facecolor='#1a1a2e', labelcolor='white')
    ax2.grid(alpha=0.15, color='gray')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Future forecast
    st.markdown(f"#### 🔮 Next {future_days} Days Forecast")
    future_prices = predict_future(model, close_vals.flatten(), scaler, seq_len, future_days)
    future_dates = pd.bdate_range(start=df.index[-1], periods=future_days + 1)[1:]

    last_price = float(close_vals[-1])
    final_price = float(future_prices[-1])
    total_change = ((final_price - last_price) / last_price) * 100

    fc1, fc2, fc3 = st.columns(3)

    fc1.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>${last_price:.2f}</div>
        <div class='metric-label'>Current Price</div>
    </div>""", unsafe_allow_html=True)

    fc2.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>${final_price:.2f}</div>
        <div class='metric-label'>Day {future_days} Forecast</div>
    </div>""", unsafe_allow_html=True)

    color_chg = 'positive' if total_change >= 0 else 'negative'
    sign_chg = '+' if total_change >= 0 else ''
    fc3.markdown(f"""<div class='metric-card'>
        <div class='metric-value {color_chg}'>{sign_chg}{total_change:.2f}%</div>
        <div class='metric-label'>Expected Change</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(14, 6), facecolor='#0e1117')
    ax3.set_facecolor('#0e1117')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('#2c3e50')

    hist_prices = close_vals[-120:].flatten()

    ax3.plot(df.index[-120:], hist_prices,
             color='#3498db', lw=2, label='Historical')

    ax3.plot(future_dates, future_prices,
             color='#e74c3c', lw=2, linestyle='--',
             marker='o', markersize=4, label=f'Forecast ({future_days}d)')

    std = np.std(future_prices) * 0.5
    ax3.fill_between(
        future_dates,
        future_prices - std,
        future_prices + std,
        alpha=0.15,
        color='#e74c3c',
        label='Confidence Band'
    )

    ax3.axvline(x=df.index[-1], color='gray', linestyle=':', lw=2)
    ax3.set_title(f'{ticker} — {future_days}-Day Price Forecast',
                  color='white', fontweight='bold')
    ax3.set_ylabel('Price (USD)', color='white')
    ax3.legend(facecolor='#1a1a2e', labelcolor='white')
    ax3.grid(alpha=0.15, color='gray')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Forecast table
    st.markdown("#### 📋 Day-by-Day Forecast")
    forecast_df = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d'),
        'Predicted Price': [f'${p:.2f}' for p in future_prices],
        'Change from Today': [f'{((p - last_price) / last_price) * 100:+.2f}%'
                              for p in future_prices]
    })
    st.dataframe(forecast_df, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#95a5a6; font-size:13px;'>
    📈 Stock Price Predictor | Built by <b>Zain Abbas</b> |
    AI & Machine Learning Engineer <br>
    ⚠️ <i>This tool is for educational purposes only.
    Not financial advice.</i>
</div>
""", unsafe_allow_html=True)