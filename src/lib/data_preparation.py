
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
        data_frame = DataPreparation.clean_data(data_frame)
        data_frame = DataPreparation.remove_stopwords(data_frame)
        
        data = data_frame.melt(var_name='sentiment', value_name='tweet')
        # data['id'] = data.index
        data['id'] = range(1, len(data) + 1) 
        
        data = data[['id'] + [col for col in data.columns if col != 'id']]
        return data


    @staticmethod
    def remove_stopwords(data_frame) -> pd.DataFrame:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english')) #python -m nltk.downloader stopwords

        new_data_frame: pd.DataFrame = data_frame.copy()
        for column in new_data_frame.columns:
            new_data_frame[column] = new_data_frame[column].apply(
                lambda row: ' '.join([word for word in row.split() if word.lower() not in stop_words])
            )
        
        return new_data_frame
    
    
    @staticmethod
    def clean_data(data_frame) -> pd.DataFrame:
        import string
        import pandas as pd

        new_data_frame: pd.DataFrame = data_frame.copy()
        for column in new_data_frame.columns:
            new_data_frame[column] = new_data_frame[column].apply(
                lambda row: ''.join(
                    char for char in row if char not in string.punctuation and not char.isdigit()
                ).lower()
            )
            
        return new_data_frame

    
