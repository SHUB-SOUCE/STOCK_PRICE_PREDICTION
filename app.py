import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title('Stock Price Prediction (LSTM)')
# Corrected the typo from load_g to load_model
model = load_model('lstm_full_model.keras')

TICKER = st.text_input('Ticker', 'AAPL')
if st.button('Predict'):
    df = yf.download(TICKER, start='2015-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'), progress=False)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Simplified RSI calculation for streamlit app
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=14).mean()
    ma_down = down.rolling(window=14).mean()
    rs = ma_up / ma_down
    df['RSI14'] = 100 - (100 / (1 + rs))

    df = df.dropna()
    feature_cols = ['Close','MA20','MA50','RSI14']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols]) 
    
    X = []
    SEQUENCE_LENGTH = 60
    last_sequence = scaled[-SEQUENCE_LENGTH:]
    X.append(last_sequence)
    X = np.array(X)

    input_data = X.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
    preds = model.predict(input_data)
    
    placeholder = np.zeros((1, len(feature_cols)))
    placeholder[0,0] = preds[0,0]
    pred_price = scaler.inverse_transform(placeholder)[0,0]
    
    st.write(f'Predicted next day close price for {TICKER}: ${pred_price:.2f}')