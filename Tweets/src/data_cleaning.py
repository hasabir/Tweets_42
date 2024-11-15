
import pandas as pd


class DataCleaning:

    @staticmethod
    def load_data():
        file_paths = {
            "positive": '../data/raw/processedPositive.csv',
            "negative": '../data/raw/processedNegative.csv',
            "neutral": '../data/raw/processedNeutral.csv'
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
    
    

if __name__ == '__main__':

    data_frame = DataCleaning.load_data()
    data_set = DataCleaning.remove_stopwords(data_frame)
    
    for column in data_frame.columns:
        for row in data_frame[column]:
            print(f"row before: {row}")
            break
        break
    for column in data_set.columns:
        for row in data_set[column]:
            print(f"row after: {row}")
            break
        break
    
    print(f"type of data frame : {type(data_frame)}, type of data set : {type(data_set)}")
    
