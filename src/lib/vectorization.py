import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Vectorization:
    @staticmethod
        
    @staticmethod
    def vectorize_with_tfidf(processed_data):
        vectorizer = TfidfVectorizer()
        text_data = processed_data['processed_tweet'].apply(lambda row: ' '.join(row))
        return vectorizer.fit_transform(text_data)
    
    @staticmethod
    def vectorize_with_bow(processed_data):
        text_data = processed_data['processed_tweet'].apply(lambda row: ' '.join(row))
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(text_data)
    
    @staticmethod
    def vectorize_with_binary_count(processed_data):
        vectorizer = CountVectorizer(binary=True)
        text_data = processed_data['processed_tweet'].apply(lambda row: ' '.join(row))
        return vectorizer.fit_transform(text_data)