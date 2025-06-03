import pytest
import pandas as pd
from src import data_utils
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_news_df():
    return pd.DataFrame({
        'headline': ['Apple rises', 'Stock falls'],
        'publisher': ['Reuters', 'Bloomberg'],
        'date': ['2020-01-01', '2020-01-02'],
        'stock': ['AAPL', 'AAPL']
    })

@pytest.fixture
def sample_stock_df():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-01-02'],
        'stock': ['AAPL', 'AAPL'],
        'open': [300, 305],
        'close': [310, 300],
        'high': [315, 310],
        'low': [295, 295],
        'volume': [1000000, 1200000]
    })

def test_load_and_validate_dataset(tmp_path):
    # Create a temporary CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({'a': [1], 'b': [2]})
    df.to_csv(csv_path, index=False)

    result = data_utils.load_and_validate_dataset(str(csv_path), expected_columns=['a', 'b'])
    assert result is not None
    assert 'a' in result.columns

def test_calculate_headline_length(sample_news_df):
    df = data_utils.calculate_headline_length(sample_news_df, 'headline')
    assert 'headline_char_length' in df.columns
    assert 'headline_word_length' in df.columns
    assert df['headline_word_length'].iloc[0] == 2

def test_count_articles_per_publisher(sample_news_df):
    counts = data_utils.count_articles_per_publisher(sample_news_df)
    assert counts['Reuters'] == 1
    assert counts['Bloomberg'] == 1

def test_analyze_publication_dates(sample_news_df):
    counts = data_utils.analyze_publication_dates(sample_news_df, 'date')
    assert counts.loc['2020-01-01'] == 1

def test_clean_and_prepare_news_data(sample_news_df):
    required_columns = ['headline', 'publisher', 'date', 'stock']
    cleaned = data_utils.clean_and_prepare_news_data(
        sample_news_df, 
        text_column='headline',
        publisher_column='publisher',
        date_column='date',
        stock_column='stock',
        required_columns=required_columns
    )
    assert 'date_only' in cleaned.columns
    assert cleaned['headline'].iloc[0] == 'Apple rises'

def test_clean_and_prepare_stock_data(sample_stock_df):
    price_cols = ['open', 'high', 'low', 'close']
    cleaned = data_utils.clean_and_prepare_stock_data(
        sample_stock_df,
        date_column='date',
        stock_column='stock',
        price_columns=price_cols,
        volume_column='volume'
    )
    assert cleaned['date'].dtype == 'datetime64[ns]'
    assert cleaned['volume'].dtype in ['int64', 'float64']
