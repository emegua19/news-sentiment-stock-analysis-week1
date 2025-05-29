# src/nlp_utils.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

# Download NLTK resources
def download_nltk_resources():
    """
    Download required NLTK resources with error handling.
    """
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        raise

download_nltk_resources()

def extract_keywords(df, text_column='headline', top_n=20):
    """
    Extract top keywords/phrases using CountVectorizer and NLTK.
    Returns DataFrame with keywords and frequencies.
    """
    # Preprocess text
    stop_words = set(stopwords.words('english')).union({'apple', 'aapl', 'stock', 'inc'})
    
    def preprocess_text(text):
        if pd.isna(text):
            return ''
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        try:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Tokenization error: {e}")
            return text
    
    texts = df[text_column].apply(preprocess_text)
    
    # Vectorize
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    
    # Get feature names and frequencies
    feature_names = vectorizer.get_feature_names_out()
    frequencies = X.sum(axis=0).A1
    keyword_df = pd.DataFrame({'Keyword/Phrase': feature_names, 'Frequency': frequencies})
    
    # Sort and return top N
    return keyword_df.sort_values(by='Frequency', ascending=False).head(top_n).reset_index(drop=True)

def run_topic_modeling(df, text_column='headline', num_topics=5):
    """
    Perform topic modeling using LDA.
    Returns list of topics and DataFrame with topic assignments.
    """
    # Preprocess text
    stop_words = set(stopwords.words('english')).union({'apple', 'aapl', 'stock', 'inc'})
    
    def preprocess_text(text):
        if pd.isna(text):
            return ''
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        try:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Tokenization error: {e}")
            return text
    
    texts = df[text_column].apply(preprocess_text)
    
    # Vectorize
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    # Run LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    # Extract topics
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx+1}: {', '.join(top_words)}")
    
    # Assign topics to documents
    topic_assignments = lda.transform(X)
    dominant_topics = topic_assignments.argmax(axis=1)
    topic_df = df[[text_column]].copy()
    topic_df['Dominant_Topic'] = dominant_topics
    topic_df['Topic_Probability'] = topic_assignments.max(axis=1)
    
    return topics, topic_df