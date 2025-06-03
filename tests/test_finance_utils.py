import pytest
import pandas as pd
import numpy as np
from src import finance_utils
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_stock_df():
    # At least 30 rows for technical indicators
    dates = pd.date_range(start="2020-01-01", periods=30)
    return pd.DataFrame({
        'Date': dates,
        'Open': np.linspace(100, 130, 30),
        'High': np.linspace(101, 131, 30),
        'Low': np.linspace(99, 129, 30),
        'Close': np.linspace(100, 130, 30),
        'Volume': np.random.randint(1000, 10000, 30)
    })

def test_load_stock_data(tmp_path):
    path = tmp_path / "stock.csv"
    df = pd.DataFrame({
        'custom_date': ['2020-01-01', '2020-01-02'],
        'Open': [100, 101],
        'High': [102, 103],
        'Low': [98, 99],
        'Close': [101, 100],
        'Volume': [1000, 1200]
    })
    df.to_csv(path, index=False)
    result = finance_utils.load_stock_data(str(path))
    assert result is not None
    assert 'Date' in result.columns
    assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(result['Date']))

def test_compute_technical_indicators(sample_stock_df):
    df = finance_utils.compute_technical_indicators(sample_stock_df)
    assert 'SMA_20' in df.columns
    assert 'RSI_14' in df.columns
    assert 'MACD' in df.columns
    assert 'BB_Upper' in df.columns
    assert 'ADX' in df.columns
    assert 'Stoch_K' in df.columns

def test_compute_financial_metrics(sample_stock_df):
    df = finance_utils.compute_financial_metrics(sample_stock_df)
    assert 'Returns' in df.columns
    assert 'Volatility' in df.columns
    assert 'Sharpe_Rolling' in df.columns
    assert 'Sharpe' in df.columns
    assert 'Cum_Returns' in df.columns
