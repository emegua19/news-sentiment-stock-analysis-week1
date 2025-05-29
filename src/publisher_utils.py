# src/publisher_utils.py

import pandas as pd
import re

def extract_email_domain(df, publisher_column='publisher'):
    """
    Extract email domains from publisher names.
    Adds 'email_domain' column to DataFrame.
    """
    def get_domain(publisher):
        if pd.isna(publisher):
            return 'unknown'
        if '@' in str(publisher):
            return publisher.split('@')[-1].lower()
        return 'non_email'
    
    df['email_domain'] = df[publisher_column].apply(get_domain)
    return df

def categorize_news_type(df, text_column='headline'):
    """
    Categorize news articles based on headline content.
    Adds 'news_type' column to DataFrame.
    """
    def assign_type(headline):
        if pd.isna(headline):
            return 'other'
        headline = headline.lower()
        if any(keyword in headline for keyword in ['earnings', 'revenue', 'profit', 'quarterly', 'financial']):
            return 'earnings'
        elif any(keyword in headline for keyword in ['price target', 'analyst', 'rating', 'upgrade', 'downgrade']):
            return 'analyst'
        elif any(keyword in headline for keyword in ['product', 'launch', 'release', 'innovation', 'device']):
            return 'product'
        elif any(keyword in headline for keyword in ['market', 'stock', 'shares', 'trading', 'volatility']):
            return 'market'
        else:
            return 'other'
    
    df['news_type'] = df[text_column].apply(assign_type)
    return df