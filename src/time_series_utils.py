import pandas as pd

def get_publication_frequency(df, date_column='date_only'):
    """
    Calculate publication frequency of news articles over time.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news data.
        date_column (str): Column name containing the date (default: 'date_only').
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Date', 'Article_Count'].
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame.")
    
    if df[date_column].dropna().empty:
        raise ValueError(f"Column '{date_column}' contains no valid data after dropping NaN values.")
    
    # Ensure datetime format
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Calculate frequency
    freq = df[date_column].value_counts().sort_index()
    freq_df = pd.DataFrame({'Date': freq.index, 'Article_Count': freq.values})
    return freq_df

def extract_publication_hour(df, date_column='date'):
    """
    Extract publication hour from a datetime column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the datetime data.
        date_column (str): Column name containing the datetime (default: 'date').
    
    Returns:
        pd.DataFrame: DataFrame with a new 'publication_hour' column.
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame.")
    
    if df[date_column].dropna().empty:
        raise ValueError(f"Column '{date_column}' contains no valid data after dropping NaN values.")
    
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], utc=True)
    df['publication_hour'] = df[date_column].dt.hour
    return df