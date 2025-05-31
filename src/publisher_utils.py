import pandas as pd
import re
from urllib.parse import urlparse

def extract_email_domain(df, publisher_column='publisher', domain_column='publisher_domain'):
    """
    Extract email domains from publisher names or a pre-derived domain column.
    
    Args:
        df (pd.DataFrame): DataFrame containing publisher data.
        publisher_column (str): Column with publisher names (default: 'publisher').
        domain_column (str): Column with pre-extracted domains (default: 'publisher_domain').
    
    Returns:
        pd.DataFrame: DataFrame with a new 'email_domain' column.
    """
    if publisher_column not in df.columns:
        raise ValueError(f"Column '{publisher_column}' not found in DataFrame.")
    
    df = df.copy()
    
    # Initialize email_domain column
    df['email_domain'] = None
    
    # Use domain_column if available and non-null
    if domain_column in df.columns:
        df['email_domain'] = df[domain_column].apply(
            lambda x: x.lower() if pd.notnull(x) and '.' in str(x) else None
        )
    
    # Fallback: Attempt to extract domain from publisher name if email_domain is still null
    mask = df['email_domain'].isna() & df[publisher_column].notna()
    if mask.any():
        df.loc[mask, 'email_domain'] = df.loc[mask, publisher_column].apply(
            lambda x: re.sub(r'^www\.|\..*$', '', urlparse('http://' + str(x)).netloc).lower() if '.' in str(x) else 'unknown'
        )
    
    # Final fallback: Set remaining nulls to 'unknown'
    df['email_domain'] = df['email_domain'].fillna('unknown')
    
    return df

def categorize_news_type(df, text_column='headline'):
    """
    Categorize news articles based on headline content using regex keyword matching.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news data.
        text_column (str): Column with headline text (default: 'headline').
    
    Returns:
        pd.DataFrame: DataFrame with a new 'news_type' column.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")
    
    df = df.copy()
    
    keywords = {
        'earnings|revenue|profit|quarterly|financial': 'earnings',
        'price target|analyst|rating|upgrade|downgrade': 'analyst',
        'product|launch|release|innovation|device': 'product',
        'market|stock|shares|trading|volatility': 'market',
        '.*': 'other'  # Catch-all for unmatched headlines
    }
    
    def assign_type(headline):
        if pd.isna(headline):
            return 'other'
        headline = headline.lower()
        for pattern, category in keywords.items():
            if re.search(pattern, headline):
                return category
        return 'other'
    
    df['news_type'] = df[text_column].apply(assign_type)
    return df