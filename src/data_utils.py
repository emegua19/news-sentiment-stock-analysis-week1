import pandas as pd
import numpy as np

def load_and_validate_dataset(file_path, expected_columns):
    """
    Load a dataset and validate that expected columns are present.
    """
    try:
        df = pd.read_csv(file_path)
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {file_path}: {missing_cols}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_headline_length(df, text_column):
    """
    Calculate character and word lengths for headlines.
    """
    df = df.copy()
    df['headline_char_length'] = df[text_column].str.len()
    df['headline_word_length'] = df[text_column].str.split().str.len()
    return df

def count_articles_per_publisher(df, publisher_column='publisher'):
    """
    Count the number of articles per publisher.
    """
    return df[publisher_column].value_counts()

def analyze_publication_dates(df, date_column):
    """
    Analyze publication trends by counting articles per date.
    """
    return df[date_column].value_counts().sort_index()

def clean_and_prepare_news_data(df, text_column, publisher_column, date_column, stock_column, required_columns):
    """
    Clean and prepare news data by handling missing values, deduplicating, and normalizing dates.
    """
    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    
    df = df.dropna(subset=required_columns)
    df = df.drop_duplicates(subset=[text_column, date_column])
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df['date_only'] = df[date_column].dt.date
    df[text_column] = df[text_column].str.strip()
    
    return df

def clean_and_prepare_stock_data(df, date_column, stock_column, price_columns, volume_column):
    """
    Clean and prepare stock data by handling missing values, ensuring numeric types, and normalizing dates.
    """
    df = df.copy()
    required_cols = [date_column, stock_column] + price_columns + [volume_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    df = df.dropna(subset=required_cols)
    for col in price_columns + [volume_column]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.sort_values(date_column)
    
    return df