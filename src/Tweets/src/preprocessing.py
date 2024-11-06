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


def tokenization(data_frame):
    tokenizer = tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = []
    for column in data_frame.columns:
        data = data_frame[column].fillna('').astype(str)
        for text in data:
            tokens.append(tokenizer.tokenize(text))
    return tokens



data_frame = load_data()
tokens_positive = tokenization(data_frame)["positive"]
print(tokens_positive)