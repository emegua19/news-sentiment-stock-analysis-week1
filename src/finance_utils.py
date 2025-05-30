# src/finance_utils.py

import pandas as pd
import numpy as np
import talib

def load_stock_data(file_path, expected_columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']):
    """
    Load stock data, validate columns, and convert numeric columns to float64.
    References: https://pandas.pydata.org/docs/user_guide/io.html#csv-text-files
    """
    try:
        df = pd.read_csv(file_path)
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        # Convert numeric columns to float64, handling errors
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('Date')
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compute_technical_indicators(df, price_column='Close', high_column='High', low_column='Low'):
    """
    Compute TA-Lib indicators: SMA, RSI, MACD, Bollinger Bands, ADX, Stochastic Oscillator.
    References: https://mrjbq7.github.io/ta-lib/, https://www.investopedia.com/terms/t/technicalindicator.asp
    """
    df = df.copy()
    # Ensure float64 for TA-Lib
    df[price_column] = df[price_column].astype(np.float64)
    df[high_column] = df[high_column].astype(np.float64)
    df[low_column] = df[low_column].astype(np.float64)
    
    # Simple Moving Average (SMA-20)
    df['SMA_20'] = talib.SMA(df[price_column], timeperiod=20)
    
    # Relative Strength Index (RSI-14)
    df['RSI_14'] = talib.RSI(df[price_column], timeperiod=14)
    
    # Moving Average Convergence Divergence (MACD)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
        df[price_column], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
        df[price_column], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    
    # Average Directional Index (ADX)
    df['ADX'] = talib.ADX(df[high_column], df[low_column], df[price_column], timeperiod=14)
    
    # Stochastic Oscillator
    df['Stoch_K'], df['Stoch_D'] = talib.STOCH(
        df[high_column], df[low_column], df[price_column],
        fastk_period=14, slowk_period=3, slowd_period=3
    )
    
    return df

def compute_financial_metrics(df, price_column='Close'):
    """
    Compute financial metrics: returns, volatility, Sharpe ratio, cumulative returns.
    References: https://www.investopedia.com/terms/s/sharperatio.asp
    """
    df = df.copy()
    # Daily log returns (manual calculation)
    df['Returns'] = np.log(df[price_column] / df[price_column].shift(1))
    # Annualized volatility (20-day rolling)
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    # Sharpe ratio (risk-free rate = 0 for simplicity)
    df['Sharpe'] = (df['Returns'].rolling(window=20).mean() * 252) / df['Volatility']
    # Cumulative returns
    df['Cum_Returns'] = (1 + df['Returns']).cumprod() - 1
    return df