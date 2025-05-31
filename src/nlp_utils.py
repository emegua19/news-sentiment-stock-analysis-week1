import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Ensure NLTK resources are available
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    raise

def extract_keywords(df, text_column, top_n=10, n_gram_range=(1, 2), custom_stop_words=None):
    """
    Extract top keywords/phrases from text using NLTK with lemmatization and n-gram support.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Column name containing the text to analyze.
        top_n (int): Number of top keywords/phrases to return.
        n_gram_range (tuple): Range of n-grams to extract (e.g., (1, 2) for unigrams and bigrams).
        custom_stop_words (list): Additional stop words to exclude.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Keyword/Phrase', 'Frequency'].
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")
    
    if df[text_column].dropna().empty:
        raise ValueError("Text column contains no valid data after dropping NaN values.")
    
    # Initialize stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    if custom_stop_words:
        stop_words.update(custom_stop_words)
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize, lemmatize, and filter
    words = []
    for text in df[text_column].dropna():
        tokens = word_tokenize(text.lower())
        filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens 
                          if token.isalnum() and token not in stop_words]
        words.extend(filtered_tokens)
    
    # Extract n-grams if needed
    if n_gram_range != (1, 1):
        vectorizer = CountVectorizer(ngram_range=n_gram_range, stop_words='english')
        ngram_matrix = vectorizer.fit_transform(df[text_column].dropna())
        ngram_counts = Counter(dict(zip(vectorizer.get_feature_names_out(), ngram_matrix.sum(axis=0).tolist()[0])))
        return pd.DataFrame(ngram_counts.most_common(top_n), columns=['Keyword/Phrase', 'Frequency'])
    
    # Otherwise, return unigram counts
    word_counts = Counter(words).most_common(top_n)
    return pd.DataFrame(word_counts, columns=['Keyword/Phrase', 'Frequency'])

def run_topic_modeling(df, text_column, num_topics=5, max_features=1000, max_iter=10, learning_method='online'):
    """
    Perform topic modeling using LDA with customizable parameters.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Column name containing the text to analyze.
        num_topics (int): Number of topics to extract.
        max_features (int): Maximum number of features for the vectorizer.
        max_iter (int): Maximum iterations for LDA.
        learning_method (str): LDA learning method ('online' or 'batch').
    
    Returns:
        tuple: (list of topics, DataFrame with topic assignments and probabilities).
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")
    
    texts = df[text_column].dropna()
    if len(texts) < num_topics:
        raise ValueError(f"Number of documents ({len(texts)}) is less than the number of topics ({num_topics}).")
    
    # Vectorize the text
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    
    # Run LDA
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=max_iter,
        learning_method=learning_method,
        random_state=42
    )
    lda.fit(dtm)
    
    # Extract topics
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics.append(" ".join(top_words))
    
    # Assign dominant topic and probabilities to each document
    topic_probs = lda.transform(dtm)
    dominant_topics = topic_probs.argmax(axis=1)
    topic_df = df[[text_column]].copy().iloc[texts.index]  # Align indices with non-NaN texts
    topic_df['Dominant_Topic'] = dominant_topics
    topic_df[[f'Topic_{i}_Prob' for i in range(num_topics)]] = topic_probs
    
    return topics, topic_df