import pandas as pd



class Vectorization:

    @staticmethod
    def _generate_word_frequencies(tokens):
        word_frequencies = {}
        for token_list in tokens:
            for token in token_list:
                word_frequencies[token] = word_frequencies.get(token, 0) + 1
        return word_frequencies
    
    @staticmethod
    def bag_of_words(tokens_positive, tokens_negative, tokens_neutral) -> dict:
        positive_bow = pd.DataFrame.from_dict(Vectorization._generate_word_frequencies(tokens_positive), orient='index', columns=['Positive']).T
        negative_bow = pd.DataFrame.from_dict(Vectorization._generate_word_frequencies(tokens_negative), orient='index', columns=['Negative']).T
        neutral_bow = pd.DataFrame.from_dict(Vectorization._generate_word_frequencies(tokens_neutral), orient='index', columns=['Neutral']).T

        bow_vectors = pd.concat([positive_bow, negative_bow, neutral_bow], axis=0)
        bow_vectors.fillna(0, inplace=True)

        return bow_vectors
    
    @staticmethod
    def tf_idf(data_set):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorized = TfidfVectorizer()
        Transformed_output = vectorized.fit_transform(data_set)
        feature_names = vectorized.get_feature_names_out()
        return Transformed_output
    
    @staticmethod
    def _binary_vectorization(tokens):
        word_presence = {}
        for token_list in tokens:
            for token in token_list:
                if token not in word_presence:
                    word_presence[token] = 1
        return word_presence

    @staticmethod
    def create_binary_vectors(positive_tokens, negative_tokens, neutral_tokens):
        positive_vectors = pd.DataFrame.from_dict(Vectorization._binary_vectorization(positive_tokens), orient='index', columns=['Positive']).T
        negative_vectors = pd.DataFrame.from_dict(Vectorization._binary_vectorization(negative_tokens), orient='index', columns=['Negative']).T
        neutral_vectors = pd.DataFrame.from_dict(Vectorization._binary_vectorization(neutral_tokens), orient='index', columns=['Neutral']).T

        combined_vectors = pd.concat([positive_vectors, negative_vectors, neutral_vectors], axis=0)
        combined_vectors.fillna(0, inplace=True)

        return combined_vectors


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

    from src.preprocessing import Preprocessing
    from src.data_cleaning import DataCleaning


    data_set = DataCleaning.load_data()
    
    tokens_positive = Preprocessing.tokenization(data_set["positive"])
    tokens_negative = Preprocessing.tokenization(data_set["negative"])
    tokens_neutral = Preprocessing.tokenization(data_set["neutral"])

    binary_vectors = binary_vectorizer(tokens_positive, tokens_negative, tokens_neutral)