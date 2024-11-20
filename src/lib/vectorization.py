import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



class Vectorization:
    @staticmethod
    def _vectorize(processed_data, vectorizer):
        text_data = processed_data.fillna('').apply(lambda row: ' '.join([' '.join(words) for words in row]), axis=1)

        vectorizer.fit(text_data)

        transformed_output = vectorizer.transform(text_data)
        feature_names = vectorizer.get_feature_names_out()
        dense_output = transformed_output.todense()   
        
        return pd.DataFrame(
                    dense_output, 
                    columns=feature_names,
                    index=text_data.index 
                )
        
    @staticmethod
    def vectorize_with_tfidf(data_frame):
        vectorizer = TfidfVectorizer()
        return Vectorization._vectorize(data_frame, vectorizer)
    
    @staticmethod
    def vectorize_with_bow(data_frame):
        vectorizer = CountVectorizer()
        return Vectorization._vectorize(data_frame, vectorizer)
    
    @staticmethod
    def vectorize_with_binary_bow(data_frame):
        vectorizer = CountVectorizer(binary=True)
        return Vectorization._vectorize(data_frame, vectorizer)