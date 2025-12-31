import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import matplotlib.pyplot as plt

def add_technical_indicators(df):
    df = df.copy()
    df['SMA'] = ta.sma(df['Close'], length=20)
    df['EMA'] = ta.ema(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    df['BB_Width'] = bb['BBB_20_2.0_2.0']
    df['BB_Percentage'] = bb['BBP_20_2.0_2.0']
    df['label'] = df['Close'].shift(-1)
    return df.dropna()

def create_sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :-1])
        y.append(data[i + window_size, -1])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
