import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import streamlit as st

def generate_text_vectors(df, text_col, method='bow', max_features=50):
    """
    Generates Bag of Words or TF-IDF vectors for a text column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_col (str): Column name containing text
        method (str): 'bow' (Bag of Words) or 'tfidf' (TF-IDF)
        max_features (int): Maximum number of features (vocabulary size) to keep
        
    Returns:
        tuple: (DataFrame with single vector column, [new_column_name]) or (DataFrame, error_message)
    """
    try:
        # Handle missing values
        texts = df[text_col].fillna("").astype(str)

        if method == 'bow':
            vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
            prefix = "bow"
        elif method == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            prefix = "tfidf"
        else:
            return df, "Invalid method selected."

        # Fit transform (sparse matrix)
        vectors = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        if len(feature_names) == 0:
            return df, "No features generated. The text column might be empty or contain only stop words."

        # Convert each row vector to a Python list and store in a single column
        dense = vectors.toarray()
        col_name = f"{text_col}_{prefix}_vector"
        result_df = df.copy()
        result_df[col_name] = [tuple(row.tolist()) for row in dense]

        return result_df, [col_name]

    except Exception as e:
        return df, str(e)
