
import pandas as pd


class DataPreparation:

    @staticmethod
    def load_data():
        file_paths = {
            "positive": '../data/raw_data/processedPositive.csv',
            "negative": '../data/raw_data/processedNegative.csv',
            "neutral": '../data/raw_data/processedNeutral.csv'
        }

        data_frame = pd.DataFrame()
        for label, path in file_paths.items():
            data_frame[label] = pd.read_csv(path, header=None).T.squeeze()

        data_frame = data_frame.fillna('').astype(str)
        data_frame = DataPreparation.clean_data(data_frame)
        # data_frame = DataPreparation.remove_stopwords(data_frame)
        
        data = data_frame.melt(var_name='sentiment', value_name='tweet')
        data['id'] = range(1, len(data) + 1) 
        
        data = data[['id'] + [col for col in data.columns if col != 'id']]
        data = DataPreparation._update_labels(data)
        return data



    
    
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
    @staticmethod
    def _update_labels(data_set: pd.DataFrame) -> pd.DataFrame:
        for index, row in data_set.iterrows():
            if row["sentiment"] == "positive":
                data_set.at[index, "label"] = 1
            elif row["sentiment"] == "negative":
                data_set.at[index, "label"] = -1
            elif row["sentiment"] == "neutral":
                data_set.at[index, "label"] = 0
            else:
                data_set.at[index, "label"] = None  # If there's an unknown sentiment
        return data_set

    