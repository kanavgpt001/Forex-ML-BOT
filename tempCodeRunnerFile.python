import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import time
import joblib
from datetime import datetime, timedelta, timezone
from IPython.display import clear_output, display
from sklearn.ensemble import RandomForestClassifier
import sys

# Ensure UTF-8 encoding for printing signals
sys.stdout.reconfigure(encoding='utf-8')

def get_eurcad_data(interval="5m", period="1d"):
    try:
        eurcad = yf.Ticker("EURCAD=X")
        data = eurcad.history(period=period, interval=interval)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.rolling(window=window).mean()
    ema_down = down.rolling(window=window).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    sma = calculate_sma(data, window)
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def extract_features(data):
    data['SMA_9'] = calculate_sma(data, 9)
    data['SMA_18'] = calculate_sma(data, 18)
    data['RSI'] = calculate_rsi(data, 14)
    data['MACD'], data['MACD_Signal'] = calculate_macd(data)
    data['Upper_BB'], data['Lower_BB'] = calculate_bollinger_bands(data)
    data['Price_Change'] = data['Close'].diff()
    return data.dropna()

def load_model():
    try:
        model = joblib.load("forex_model.pkl")
        return model
    except:
        print("Model not found. Train and save a model first.")
        return None

def predict_signal(data, model):
    if model is None or data is None or data.empty:
        return None, None
    
    features = ['SMA_9', 'SMA_18', 'RSI', 'MACD', 'MACD_Signal', 'Upper_BB', 'Lower_BB', 'Price_Change']
    latest_data = data[features].iloc[-1:].values.reshape(1, -1)
    prediction = model.predict(latest_data)
    
    signal_mapping = {1: "CALL 🟩", -1: "PUT 🟥", 0: "HOLD"}
    signal_type = signal_mapping.get(prediction[0], "HOLD")
    
    current_time = datetime.now(timezone.utc) - timedelta(hours=3)  # UTC -3
    trade_time = (current_time + timedelta(minutes=5)).strftime("%H:%M")
    
    if signal_type != "HOLD":
        signal_text = f"💰 5 minutes expiry\nEUR/CAD;{trade_time};{signal_type}\n🕐 TIME TO {trade_time}"
        return signal_text, trade_time
    
    return None, None

def create_candlestick_chart(data, title):
    if data is None or data.empty:
        print(f"No data to display for {title}.")
        return None
    
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False)
    return fig

def update_charts_and_signals():
    model = load_model()
    signals_history = []
    
    try:
        while True:
            data_5min = get_eurcad_data(interval="5m")
            data_5min = extract_features(data_5min)
            fig_5min = create_candlestick_chart(data_5min, "EUR/CAD 5-Minute Chart")
            
            if fig_5min is not None:
                signal_text, trade_time = predict_signal(data_5min, model)
                
                if signal_text:
                    signals_history.append(signal_text)
                    signals_history = signals_history[-5:]
                    clear_output(wait=True)
                    display(fig_5min)
                    print("Trading Signals:")
                    for signal in signals_history:
                        print(signal)
                else:
                    clear_output(wait=True)
                    display(fig_5min)
                    print("No New Trading Signals")
            
            time.sleep(600)  # 10-minute interval
    except KeyboardInterrupt:
        print("Chart update stopped.")

if __name__ == "__main__":
    update_charts_and_signals()
