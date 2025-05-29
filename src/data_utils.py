# src/data_utils.py

import pandas as pd
import numpy as np
import re

def load_and_validate_dataset(file_path, expected_columns=None):
    """
    Load a dataset and validate expected columns.
    """
    try:
        df = pd.read_csv(file_path)
        if expected_columns:
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing expected columns: {missing_cols}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def clean_and_prepare_news_data(df, text_column='headline', publisher_column='publisher', 
                               date_column='date', stock_column='stock', required_columns=['headline', 'stock']):
    """
    Clean and prepare news dataset.
    """
    df = df.dropna(subset=required_columns)
    
    def clean_text(text):
        if pd.isna(text):
            return text
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    df[text_column] = df[text_column].apply(clean_text)
    
    def get_domain(publisher):
        if pd.isna(publisher):
            return publisher
        if '@' in publisher:
            return publisher.split('@')[-1]
        return publisher
    
    df['publisher_domain'] = df[publisher_column].apply(get_domain)
    
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, utc=True)
        except:
            return pd.NaT
    
    df[date_column] = df[date_column].apply(parse_date)
    df = df.dropna(subset=[date_column])
    df['date_only'] = df[date_column].dt.date
    
    df[stock_column] = df[stock_column].str.upper()
    
    return df

def clean_and_prepare_stock_data(df, date_column='Date', stock_column='stock', 
                                price_columns=['Open', 'High', 'Low', 'Close'], volume_column='Volume'):
    """
    Clean and prepare stock dataset.
    """
    df = df.dropna(subset=price_columns + [volume_column])
    
    df[date_column] = pd.to_datetime(df[date_column], utc=True)
    df['date_only'] = df[date_column].dt.date
    
    df = df[df[price_columns].gt(0).all(axis=1)]
    df = df[df[volume_column] >= 0]
    
    return df

def calculate_headline_length(df, text_column='headline'):
    """
    Calculate character and word length of headlines.
    """
    df['headline_char_length'] = df[text_column].str.len()
    df['headline_word_length'] = df[text_column].str.split().str.len()
    return df

def count_articles_per_publisher(df, publisher_column='publisher_domain'):
    """
    Count articles per publisher.
    """
    return df[publisher_column].value_counts()

def analyze_publication_dates(df, date_column='date_only'):
    """
    Analyze publication date trends.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    return df[date_column].value_counts().sort_index()