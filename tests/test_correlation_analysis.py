import pytest 
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.correlation_analysis import (
    normalize_to_date_column,
    perform_sentiment_analysis,
    calculate_stock_returns,
    align_data,
    calculate_correlation,
)

# Sample news data
news_sample_raw = pd.DataFrame({
    'date': ['2020-01-01 10:00:00', '2020-01-02 11:30:00', '2020-01-03 09:45:00'],
    'headline': ['AAPL soars after earnings', 'Investors cautious on AAPL', 'AAPL outlook remains steady']
})

# Sample stock data
stock_sample = pd.DataFrame({
    'Date': ['2020-01-01', '2020-01-02', '2020-01-03'],
    'Close': [300.0, 305.0, 310.0]
})

def test_normalize_to_date_column():
    df = normalize_to_date_column(news_sample_raw, 'date')
    assert 'Date' in df.columns
    assert pd.api.types.is_object_dtype(df['Date'])

def test_perform_sentiment_analysis():
    news_df = normalize_to_date_column(news_sample_raw, 'date')
    df, daily_sentiment = perform_sentiment_analysis(news_df)
    assert 'Sentiment' in df.columns
    assert 'Tone' in df.columns
    assert daily_sentiment.shape[0] == 3
    assert set(daily_sentiment.columns) == {'Date', 'Sentiment', 'Tone'}

def test_calculate_stock_returns():
    df = calculate_stock_returns(stock_sample)
    assert 'Returns' in df.columns
    assert df['Returns'].notnull().all()

def test_align_data():
    news_df = normalize_to_date_column(news_sample_raw, 'date')
    _, sentiment_df = perform_sentiment_analysis(news_df)
    stock_df = calculate_stock_returns(stock_sample)
    aligned = align_data(sentiment_df, stock_df)
    assert all(col in aligned.columns for col in ['Date', 'Sentiment', 'Tone', 'Returns'])

def test_calculate_correlation():
    news_df = normalize_to_date_column(news_sample_raw, 'date')
    _, sentiment_df = perform_sentiment_analysis(news_df)
    stock_df = calculate_stock_returns(stock_sample)
    aligned = align_data(sentiment_df, stock_df)
    r, p = calculate_correlation(aligned)
    assert isinstance(r, float) or np.isnan(r)
    assert isinstance(p, float) or np.isnan(p)

def test_calculate_correlation_with_lag():
    news_df = normalize_to_date_column(news_sample_raw, 'date')
    _, sentiment_df = perform_sentiment_analysis(news_df)
    stock_df = calculate_stock_returns(stock_sample)
    aligned = align_data(sentiment_df, stock_df)
    r, p = calculate_correlation(aligned, lag=1)
    assert isinstance(r, float) or np.isnan(r)
    assert isinstance(p, float) or np.isnan(p)
