# src/time_series_utils.py

import pandas as pd

def get_publication_frequency(df, date_column='date_only'):
    """
    Calculate daily publication frequency of news articles.
    Returns DataFrame with dates and article counts.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    freq = df[date_column].value_counts().sort_index()
    freq_df = pd.DataFrame({'Date': freq.index, 'Article_Count': freq.values})
    return freq_df

def extract_publication_hour(df, date_column='date'):
    """
    Extract publication hour from datetime column.
    Adds 'publication_hour' column to DataFrame.
    """
    df[date_column] = pd.to_datetime(df[date_column], utc=True)
    df['publication_hour'] = df[date_column].dt.hour
    return df