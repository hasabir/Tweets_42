
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
    def remove_stopwords(data_frame) -> list:
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words('english'))
        filtered_texts = []

        for data in data_frame:
            tokens = data.split()
            filtered_texts.append(" ".join([word for word in tokens if word.lower() not in stop_words]))

        return filtered_texts
    
    
    
if __name__ == '__main__':

    DataCleaning.load_data()
    # data_set = DataCleaning.load_data()
    # for data in data_set:
    #     print(data)
    # print(data_set)