import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('ggplot')

import nltk


def load_data():
    positive = pd.DataFrame()
    negative = pd.DataFrame()
    neutral = pd.DataFrame()

    positive["positive"] = pd.read_csv('../data/raw/processedPositive.csv', header=None ).T.squeeze()
    negative["negative"] = pd.read_csv('../data/raw/processedNegative.csv', header=None).T.squeeze()
    neutral["neutral"] = pd.read_csv('../data/raw/processedNeutral.csv', header=None).T.squeeze()

    merged = pd.merge(positive, negative, left_index=True, right_index=True)
    data_frame = pd.merge(merged, neutral, left_index=True, right_index=True)
    return data_frame


def tokenization(data_frame) -> list:
    tokenizer = tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = []
    for data in data_frame:
        tokens.append(tokenizer.tokenize(data))
    return tokens




def stemming(data_frame) -> list:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    stemmed_words = []
    for data in data_frame:
        stemmed_words.append([stemmer.stem(word) for word in data])
    return stemmed_words



if __name__ == "__main__":
    data_frame = load_data()
    tokens_positive = tokenization(data_frame)["positive"]
    stemmed_words = stemming(tokens_positive)
    print(stemmed_words)
    print(tokens_positive)
