import pandas as pd



class Vectorization:

    @staticmethod
    def bag_of_words(data_frame: list) -> dict:
        bag_of_words = {}
        for data in data_frame:
            for word in data:
                if word in bag_of_words:
                    bag_of_words[word] += 1
                else:
                    bag_of_words[word] = 1
        return bag_of_words
    
    
    @staticmethod
    def tf_idf(data_set):
        pass