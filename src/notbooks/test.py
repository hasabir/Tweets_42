

import pandas as pd
import sys
import os

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

from lib.data_preparation import DataPreparation
from lib.preprocessing_data import Preprocessing
from lib.vectorization import Vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data_set = DataPreparation().load_data()
# data_set = Preprocessing().tokenization(data_set)
# data_set = Preprocessing().lemmatization(data_set)
data_set = Preprocessing().lemmatization_with_misspelling(data_set)
data_set = data_set.sample(frac=1, random_state=42).reset_index(drop=True)

split_index_1 = int(len(data_set) * 0.7)
split_index_2 = int(len(data_set) * 0.8)

train_df = data_set[:split_index_1]
test_df = data_set[split_index_1:split_index_2]
validation_df = data_set[split_index_2:]

print(len(train_df), len(test_df), len(validation_df))

X_train = train_df['processed_tweet']
y_train = train_df['label'].to_numpy().astype(int)

X_test = test_df['processed_tweet']
y_test = test_df['label'].to_numpy().astype(int)

X_validation = validation_df['processed_tweet']
y_validation = validation_df['label'].to_numpy().astype(int)

vectorizer = TfidfVectorizer()
vectorizer = CountVectorizer(binary=True)
X_train_vector = vectorizer.fit_transform(X_train if isinstance(X_train.iloc[0], str) else X_train.apply(' '.join))
X_test_vector = vectorizer.transform(X_test if isinstance(X_test.iloc[0], str) else X_test.apply(' '.join))

print(X_train_vector.shape)
print(X_test_vector.shape)

clf = LogisticRegression()
clf.fit(X_train_vector, y_train)
y_pred = clf.predict(X_test_vector)

# print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True)

# Print the classification report with 3 decimal places
formatted_report = pd.DataFrame(report).transpose()
print(formatted_report.round(3))
