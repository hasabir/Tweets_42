
import pandas as pd


class DataPreparation:

    @staticmethod
    def load_data():
        file_paths = {
            "positive": '../data/processedPositive.csv',
            "negative": '../data/processedNegative.csv',
            "neutral": '../data/processedNeutral.csv'
        }
        
        data_frame = pd.DataFrame()
        for label, path in file_paths.items():
            data_frame[label] = pd.read_csv(path, header=None).T.squeeze()

        data_frame = data_frame.fillna('').astype(str)
        return data_frame


    @staticmethod
    def remove_stopwords(data_frame) -> pd.DataFrame:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))

        new_data_frame: pd.DataFrame = data_frame.copy()
        for column in new_data_frame.columns:
            new_data_frame[column] = new_data_frame[column].apply(
                lambda row: ' '.join([word for word in row.split() if word.lower() not in stop_words])
            )
        
        return new_data_frame
    
    
    @staticmethod
    def remove_punctuation(data_frame) -> pd.DataFrame:
        import string
        
        new_data_frame: pd.DataFrame = data_frame.copy()
        for column in new_data_frame.columns:
            new_data_frame[column] = new_data_frame[column].apply(
                lambda row: row.translate(str.maketrans('', '', string.punctuation))
            )
        
        return new_data_frame
    
