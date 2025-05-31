import os
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import nltk

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon', quiet=True)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def normalize_to_date_column(df: pd.DataFrame, datetime_col: str, date_format: str = None, new_col: str = 'Date') -> pd.DataFrame:
    """
    Ensure a DataFrame has a date-only column derived from a specified datetime column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing a timestamp column.
    datetime_col : str
        The name of the column in `df` to be converted to datetime and then reduced to a date.
    date_format : str, optional
        The format of the date string if not standard (e.g., '%m/%d/%Y').
    new_col : str, default 'Date'
        The name of the new column that will contain only the date portion (YYYY-MM-DD).
    
    Returns
    -------
    pd.DataFrame
        A copy of the original DataFrame with `new_col` added.
    
    Raises
    ------
    KeyError
        If `datetime_col` is not found in `df.columns`.
    """
    df = df.copy()
    if datetime_col not in df.columns:
        raise KeyError(f"No column named '{datetime_col}' found. Available columns: {df.columns.tolist()}")

    if date_format:
        df[datetime_col] = pd.to_datetime(df[datetime_col], format=date_format, errors='coerce', utc=True)
    else:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce', utc=True)
    
    df[new_col] = df[datetime_col].dt.date
    return df

def load_and_prepare_data(news_file: str, stock_file: str, news_date_col: str = 'date', stock_date_col: str = 'Date') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare news and stock datasets, aligning dates by normalizing timestamps.

    Parameters
    ----------
    news_file : str
        Path to the news CSV file.
    stock_file : str
        Path to the stock CSV file.
    news_date_col : str, default 'date'
        Name of the date column in the news data.
    stock_date_col : str, default 'Date'
        Name of the date column in the stock data.

    Returns
    -------
    news_df : pd.DataFrame
        News DataFrame with at least columns ['Date', 'headline', ...].
    stock_df : pd.DataFrame
        Stock DataFrame with at least columns ['Date', 'Close', ...].
    """
    # Load news data
    news_df = pd.read_csv(news_file)
    if news_date_col not in news_df.columns:
        raise KeyError(f"No '{news_date_col}' in news data. Available: {news_df.columns.tolist()}")
    news_df = normalize_to_date_column(news_df, news_date_col)

    # Load stock data
    stock_df = pd.read_csv(stock_file)
    if stock_date_col not in stock_df.columns:
        raise KeyError(f"No '{stock_date_col}' in stock data. Available: {stock_df.columns.tolist()}")
    stock_df = normalize_to_date_column(stock_df, stock_date_col)

    if 'Close' not in stock_df.columns:
        raise KeyError(f"No 'Close' column in stock data. Available: {stock_df.columns.tolist()}")

    # Filter for AAPL if applicable
    if 'stock' in stock_df.columns:
        stock_df = stock_df[stock_df['stock'].str.upper() == 'AAPL']
    elif 'symbol' in stock_df.columns:
        stock_df = stock_df[stock_df['symbol'].str.upper() == 'AAPL']
    else:
        print("Warning: No 'stock' or 'symbol' column in stock data. Assuming all data is for AAPL.")

    return news_df, stock_df

def perform_sentiment_analysis(news_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conduct sentiment analysis on news headlines using VADER and categorize into 'positive', 'negative', or 'neutral'.

    Parameters
    ----------
    news_df : pd.DataFrame
        DataFrame with at least ['Date', 'headline' OR 'title'].

    Returns
    -------
    news_df : pd.DataFrame
        Original DataFrame with added 'Sentiment' and 'Tone' columns.
    daily_sentiment : pd.DataFrame
        Grouped by 'Date', contains ['Date', 'Sentiment', 'Tone'] where:
          - 'Sentiment' is average daily compound score,
          - 'Tone' is the most frequent sentiment category ('positive', 'negative', or 'neutral').
    """
    if 'headline' in news_df.columns:
        headline_col = 'headline'
    elif 'title' in news_df.columns:
        headline_col = 'title'
    else:
        raise KeyError(f"No recognizable headline column. Available: {news_df.columns.tolist()}")

    def compute_sentiment(text):
        try:
            score = sia.polarity_scores(str(text))['compound']
            if score >= 0.05:
                return score, 'positive'
            elif score <= -0.05:
                return score, 'negative'
            else:
                return score, 'neutral'
        except Exception:
            return np.nan, 'neutral'

    news_df = news_df.copy()
    news_df['Sentiment'], news_df['Tone'] = zip(*news_df[headline_col].apply(compute_sentiment))

    # Aggregate daily sentiment
    daily_sentiment = news_df.groupby('Date', as_index=False).agg({
        'Sentiment': 'mean',
        'Tone': lambda x: x.mode()[0] if not x.mode().empty else 'neutral'
    })

    return news_df, daily_sentiment

def calculate_stock_returns(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the percentage change in daily closing prices.

    Parameters
    ----------
    stock_df : pd.DataFrame
        DataFrame with at least ['Date', 'Close'].

    Returns
    -------
    stock_df : pd.DataFrame
        Original DataFrame with an added 'Returns' column (percent).
    """
    if 'Close' not in stock_df.columns:
        raise KeyError(f"No 'Close' column in stock data. Available: {stock_df.columns.tolist()}")

    stock_df = stock_df.copy()
    stock_df = stock_df.sort_values(by='Date').reset_index(drop=True)

    if stock_df['Date'].duplicated().any():
        print("Warning: Duplicate dates found in stock data. Keeping first occurrence.")
        stock_df = stock_df.drop_duplicates(subset='Date', keep='first')

    stock_df['Returns'] = stock_df['Close'].pct_change() * 100
    stock_df = stock_df.dropna(subset=['Returns']).reset_index(drop=True)
    return stock_df

def align_data(daily_sentiment: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align sentiment and stock data by 'Date' using an inner merge.

    Parameters
    ----------
    daily_sentiment : pd.DataFrame
        DataFrame with ['Date', 'Sentiment', 'Tone'].
    stock_df : pd.DataFrame
        DataFrame with ['Date', 'Returns'].

    Returns
    -------
    aligned_df : pd.DataFrame
        Contains ['Date', 'Sentiment', 'Tone', 'Returns'].
    """
    if 'Date' not in daily_sentiment.columns or 'Date' not in stock_df.columns:
        raise KeyError("Both DataFrames must contain a 'Date' column.")

    merged = pd.merge(daily_sentiment[['Date', 'Sentiment', 'Tone']], stock_df[['Date', 'Returns']], on='Date', how='inner')
    merged = merged.dropna(subset=['Sentiment', 'Returns']).reset_index(drop=True)
    return merged

def calculate_correlation(aligned_df: pd.DataFrame, lag: int = 0) -> tuple[float, float]:
    """
    Compute Pearson correlation between sentiment at t-lag and returns at t.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        DataFrame with at least ['Sentiment', 'Returns'].
    lag : int, default 0
        Number of days to lag sentiment.

    Returns
    -------
    (r_value, p_value) : tuple of floats
        Pearson correlation coefficient and p-value.
    """
    df = aligned_df.copy()
    if lag > 0:
        df['Sentiment'] = df['Sentiment'].shift(lag)

    df = df.dropna(subset=['Sentiment', 'Returns']).reset_index(drop=True)
    if df.shape[0] < 2 or df['Sentiment'].nunique() < 2 or df['Returns'].nunique() < 2:
        return np.nan, np.nan

    try:
        r_value, p_value = pearsonr(df['Sentiment'], df['Returns'])
    except Exception:
        return np.nan, np.nan

    return r_value, p_value

def plot_correlation(aligned_df: pd.DataFrame, save_path: str = None):
    """
    Visualize the relationship between sentiment and stock returns.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        DataFrame with ['Sentiment', 'Returns'].
    save_path : str, optional
        Path to save the plot.
    """
    df = aligned_df.dropna(subset=['Sentiment', 'Returns']).reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Sentiment'], df['Returns'], alpha=0.5)
    plt.title('Sentiment vs. AAPL Returns')
    plt.xlabel('Sentiment Score')
    plt.ylabel('AAPL Returns (%)')
    plt.grid(True)

    if df.shape[0] >= 2:
        m, b = np.polyfit(df['Sentiment'], df['Returns'], 1)
        plt.plot(df['Sentiment'], m * df['Sentiment'] + b, color='red', linewidth=1,
                 label=f'y = {m:.2f}x + {b:.2f}')
        plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()