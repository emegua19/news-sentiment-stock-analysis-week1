# src/correlation_analysis.py

import os
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Ensure output directories exist (for saving plots or results if needed)
os.makedirs(os.path.join(os.path.dirname(__file__), "plots"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "../data"), exist_ok=True)


def normalize_to_date_column(df: pd.DataFrame, datetime_col: str, new_col: str = 'date_only') -> pd.DataFrame:
    """
    Ensure a DataFrame has a date-only column derived from a specified datetime column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing a timestamp column.
    datetime_col : str
        The name of the column in `df` to be converted to datetime and then reduced to a date.
    new_col : str, default 'date_only'
        The name of the new column that will contain only the date portion (YYYY-MM-DD).
    
    Returns
    -------
    pd.DataFrame
        A copy of the original DataFrame, with:
          - `datetime_col` converted to pandas datetime (UTC),
          - `new_col` added, containing the date (as a Python `date`) for each row.
    
    Raises
    ------
    KeyError
        If `datetime_col` is not found in `df.columns`.
    """
    df = df.copy()
    if datetime_col not in df.columns:
        raise KeyError(f"No column named '{datetime_col}' found. Available columns: {df.columns.tolist()}")

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce', utc=True)
    df[new_col] = df[datetime_col].dt.date
    return df


def load_and_prepare_data(news_file: str, stock_file: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Load and prepare news and stock datasets, aligning dates by normalizing timestamps.

    - Reads CSV files from paths `news_file` and `stock_file`.
    - For news_df, looks for a 'date' or 'publish_date' column, converts it to date-only under 'Date'.
    - For stock_df, looks for a 'Date' or 'date_only' column, converts it to date-only under 'Date'.

    Returns
    -------
    news_df : pd.DataFrame
        News DataFrame with at least columns ['Date', 'headline', ...].
    stock_df : pd.DataFrame
        Stock DataFrame with at least columns ['Date', 'Close', ...].

    Raises
    ------
    KeyError
        If neither expected date column is found in either CSV.
    """
    # Load news data
    news_df = pd.read_csv(news_file)
    # Debug: print(news_df.columns.tolist())

    if 'date' in news_df.columns:
        news_df['Date'] = pd.to_datetime(news_df['date'], errors='coerce').dt.date
    elif 'publish_date' in news_df.columns:
        news_df['Date'] = pd.to_datetime(news_df['publish_date'], errors='coerce').dt.date
    else:
        raise KeyError(
            f"No recognizable date column in news data. Available columns: {news_df.columns.tolist()}"
        )

    # Load stock data
    stock_df = pd.read_csv(stock_file)
    # Debug: print(stock_df.columns.tolist())

    if 'Date' in stock_df.columns:
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce').dt.date
    elif 'date_only' in stock_df.columns:
        stock_df['Date'] = pd.to_datetime(stock_df['date_only'], errors='coerce').dt.date
    else:
        raise KeyError(
            f"No recognizable date column in stock data. Available columns: {stock_df.columns.tolist()}"
        )

    return news_df, stock_df


def perform_sentiment_analysis(news_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Conduct sentiment analysis on news headlines to quantify tone.

    - Expects a DataFrame `news_df` with at least ['Date', 'headline' OR 'title'].
    - Computes TextBlob polarity per headline, stores in 'Sentiment'.
    - Assigns 'Tone' per row: 'positive', 'negative', or 'neutral'.
    - Aggregates daily sentiment (mean polarity, mode of Tone) into `daily_sentiment`.

    Returns
    -------
    news_df : pd.DataFrame
        Original DataFrame with added columns ['Sentiment', 'Tone'].
    daily_sentiment : pd.DataFrame
        Grouped by 'Date', contains ['Date', 'Sentiment', 'Tone'] where:
          - 'Sentiment' is average daily polarity,
          - 'Tone' is mode of daily row-level Tone (or 'neutral' if tie).
    """
    # Identify headline column
    if 'headline' in news_df.columns:
        headline_col = 'headline'
    elif 'title' in news_df.columns:
        headline_col = 'title'
    else:
        raise KeyError(
            f"No recognizable headline column. Available columns: {news_df.columns.tolist()}"
        )

    # Compute polarity
    def compute_polarity(text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except Exception:
            return np.nan

    news_df = news_df.copy()
    news_df['Sentiment'] = news_df[headline_col].apply(compute_polarity)
    news_df['Tone'] = news_df['Sentiment'].apply(
        lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
    )

    # Aggregate by day
    daily_sentiment = (
        news_df.groupby('Date', as_index=False)
        .agg({
            'Sentiment': 'mean',
            'Tone': lambda x: x.mode()[0] if not x.mode().empty else 'neutral'
        })
    )

    return news_df, daily_sentiment


def calculate_stock_returns(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the percentage change in daily closing prices to represent stock movements.

    - Expects a DataFrame `stock_df` with at least ['Date', 'Close'].
    - Sorts by 'Date', computes percent change on 'Close', multiplies by 100.
    - Drops the first NaN row produced by pct_change.
    
    Returns
    -------
    stock_df : pd.DataFrame
        Original DataFrame with an added 'Returns' column (percent).
    
    Raises
    ------
    KeyError
        If 'Close' is not present in stock_df.
    """
    if 'Close' not in stock_df.columns:
        raise KeyError(f"No 'Close' column in stock data. Available: {stock_df.columns.tolist()}")

    stock_df = stock_df.copy()
    stock_df = stock_df.sort_values(by='Date').reset_index(drop=True)
    stock_df['Returns'] = stock_df['Close'].pct_change() * 100
    stock_df = stock_df.dropna(subset=['Returns']).reset_index(drop=True)
    return stock_df


def align_data(daily_sentiment: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align sentiment and stock data by 'Date' using an inner merge.

    - Expects `daily_sentiment` with ['Date', 'Sentiment', 'Tone'].
    - Expects `stock_df` with ['Date', 'Returns'].
    - Merges on 'Date', inner join, drops any rows with NaN in Sentiment or Returns.

    Returns
    -------
    aligned_df : pd.DataFrame
        Contains columns ['Date', 'Sentiment', 'Tone', 'Returns'] for dates present in both inputs.
    """
    merged = pd.merge(
        daily_sentiment[['Date', 'Sentiment', 'Tone']],
        stock_df[['Date', 'Returns']],
        on='Date',
        how='inner'
    )
    merged = merged.dropna(subset=['Sentiment', 'Returns']).reset_index(drop=True)
    return merged


def calculate_correlation(aligned_df: pd.DataFrame, lag: int = 0) -> (float, float):
    """
    Compute Pearson correlation between daily sentiment and daily returns.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        DataFrame with at least ['Sentiment', 'Returns'].
    lag : int, default 0
        If lag > 0, shift 'Sentiment' upward so sentiment at day t aligns with returns at day t+lag.
    
    Returns
    -------
    (r_value, p_value) : tuple of floats
        Pearson correlation coefficient and two-tailed p-value. Returns (np.nan, np.nan) if
        there are fewer than 2 valid points or if either series is constant.
    """
    df = aligned_df.copy()
    if lag > 0:
        df['Sentiment'] = df['Sentiment'].shift(-lag)

    df = df.dropna(subset=['Sentiment', 'Returns']).reset_index(drop=True)
    if df.shape[0] < 2:
        return np.nan, np.nan
    if df['Sentiment'].nunique() < 2 or df['Returns'].nunique() < 2:
        return np.nan, np.nan

    try:
        r_value, p_value = pearsonr(df['Sentiment'], df['Returns'])
    except Exception:
        return np.nan, np.nan

    return r_value, p_value


def plot_correlation(aligned_df: pd.DataFrame, save_path: str = 'plots/sentiment_vs_returns.png'):
    """
    Visualize the relationship between sentiment and stock returns as a scatter plot.

    - Drops NaNs in ['Sentiment', 'Returns'].
    - Plots points (Sentiment, Returns) and draws a best-fit line if ≥2 points.
    - Saves figure at `save_path` and also shows it.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        DataFrame containing ['Sentiment', 'Returns'].
    save_path : str, default '../plots/sentiment_vs_returns.png'
        File path to save the scatter plot image.

    Returns
    -------
    None
    """
    df = aligned_df.dropna(subset=['Sentiment', 'Returns']).reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Sentiment'], df['Returns'], alpha=0.5)
    plt.title('Sentiment vs. Stock Returns')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Stock Returns (%)')
    plt.grid(True)

    if df.shape[0] >= 2:
        m, b = np.polyfit(df['Sentiment'], df['Returns'], 1)
        plt.plot(df['Sentiment'], m * df['Sentiment'] + b, color='red', linewidth=1,
                 label=f'y = {m:.2f}x + {b:.2f}')
        plt.legend()

    plt.savefig(save_path)
    plt.show()
