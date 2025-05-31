import pandas as pd
import numpy as np
import talib

def load_stock_data(file_path, expected_columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']):
    """
    Load stock data, validate columns, and convert numeric columns to float64.
    Dynamically detects date column if 'Date' is missing.
    References: https://pandas.pydata.org/docs/user_guide/io.html#csv-text-files
    """
    try:
        df = pd.read_csv(file_path)
        # Dynamically detect date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        if not date_col:
            raise KeyError(f"No date-related column found in {file_path}. Available columns: {df.columns.tolist()}")
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce').dt.date
        
        # Validate and convert numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        df = df.sort_values('Date')
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compute_technical_indicators(df, price_column='Close', high_column='High', low_column='Low'):
    """
    Compute TA-Lib indicators: SMA, RSI, MACD, Bollinger Bands, ADX, Stochastic Oscillator.
    Allows custom column names for flexibility.
    References: https://mrjbq7.github.io/ta-lib/, https://www.investopedia.com/terms/t/technicalindicator.asp
    """
    if not all(col in df.columns for col in [price_column, high_column, low_column]):
        raise ValueError(f"Required columns missing: {price_column}, {high_column}, or {low_column} not in DataFrame.")
    
    if len(df) < 26:  # Minimum length for MACD (slowperiod=26)
        raise ValueError(f"Insufficient data points ({len(df)}). Need at least 26 rows for indicator calculations.")
    
    df = df.copy()
    
    # Ensure float64 for TA-Lib and handle NaNs
    df[price_column] = pd.to_numeric(df[price_column], errors='coerce').astype(np.float64)
    df[high_column] = pd.to_numeric(df[high_column], errors='coerce').astype(np.float64)
    df[low_column] = pd.to_numeric(df[low_column], errors='coerce').astype(np.float64)
    
    # Drop rows with NaN in required columns to avoid TA-Lib errors
    df = df.dropna(subset=[price_column, high_column, low_column])
    
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
    Allows custom price column name.

    References: https://www.investopedia.com/terms/s/sharperatio.asp
    """
    if price_column not in df.columns:
        raise ValueError(f"Column '{price_column}' not found in DataFrame.")
    
    if len(df) < 20:  # Minimum length for 20-day rolling calculations
        raise ValueError(f"Insufficient data points ({len(df)}). Need at least 20 rows for metric calculations.")
    
    df = df.copy()

    # Daily log returns
    df['Returns'] = np.log(df[price_column] / df[price_column].shift(1))
    
    # Annualized volatility (20-day rolling)
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    # Rolling Sharpe ratio (risk-free rate = 0 for simplicity)
    df['Sharpe_Rolling'] = (df['Returns'].rolling(window=20).mean() * 252) / df['Volatility']
    
    # Overall Sharpe ratio (annualized, assuming 252 trading days)
    risk_free_rate = 0.0  # Simplified for the challenge
    mean_return = df['Returns'].mean() * 252
    std_return = df['Returns'].std() * np.sqrt(252)
    df['Sharpe'] = (mean_return - risk_free_rate) / std_return if std_return != 0 else 0
    
    # Cumulative returns
    df['Cum_Returns'] = (1 + df['Returns']).cumprod() - 1
    
    return df