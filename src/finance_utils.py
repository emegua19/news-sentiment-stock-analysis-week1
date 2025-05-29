# src/finance_utils.py

import pandas as pd
import numpy as np
import talib
import pynance as pn

def load_stock_data(file_path, expected_columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']):
    """
    Load stock data into a DataFrame and validate columns.
    """
    try:
        df = pd.read_csv(file_path)
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.sort_values('Date')
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compute_technical_indicators(df, price_column='Close'):
    """
    Compute technical indicators using TA-Lib and pynance.
    Adds SMA, RSI, MACD, and Bollinger Bands to DataFrame.
    """
    # Ensure float64 for TA-Lib
    df[price_column] = df[price_column].astype(np.float64)
    
    # Simple Moving Average (SMA)
    df['SMA_20'] = talib.SMA(df[price_column], timeperiod=20)
    
    # Relative Strength Index (RSI)
    df['RSI_14'] = talib.RSI(df[price_column], timeperiod=14)
    
    # Moving Average Convergence Divergence (MACD)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df[price_column], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df[price_column], timeperiod=20)
    
    return df