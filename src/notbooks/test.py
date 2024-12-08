import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

from lib.data_preparation import DataPreparation
from lib.preprocessing_data import Preprocessing
from lib.vectorization import Vectorization


# data_set = DataPreparation().load_data()
# data_set = Preprocessing().tokenization(data_set)

# train_df = data_set.sample(frac=1, random_state=1).reset_index(drop=True, inplace=True)

# split_index_1 = int(len(data_set) * 0.7)
# split_index_2 = int(len(data_set) * 0.8)


# train_df, test_df, validation_df = data_set[:split_index_1], data_set[split_index_1:split_index_2], data_set[split_index_2:]

# sequence_lengths = []

# from sklearn.model_selection import train_test_split



# X_train = Vectorization.vectorize_with_tfidf(train_df['tweet'])
# y_train = train_df['sentiment'].to_numpy().astype(str)

# X_test = Vectorization.vectorize_with_tfidf(test_df['tweet'])
# y_test = test_df['sentiment'].to_numpy().astype(str)




# for i in range(len(X_train)):
#   sequence_lengths.append(len(X_train[i]))


# from copy import deepcopy

# import numpy as np

# def pad_X(X, desired_sequence_length=125):
#   X_copy = deepcopy(X)

#   for i, x in enumerate(X):
#     x_seq_len = x.shape[0]
#     sequence_length_difference = desired_sequence_length - x_seq_len
    
#     pad = np.zeros(shape=(sequence_length_difference, 4062))

#     X_copy[i] = np.concatenate([x, pad])
  
#   return np.array(X_copy).astype(float)

# print(f"X_train_vector shape: {X_train_vector.shape}")
# print(f"X_train shape before padding: {X_train.shape[0]}")

# X_train = pad_X(X_train)


# print(X_train.shape , y_train.shape)

# *********************************************************************************


# from sklearn.model_selection import train_test_split
# import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import nltk
# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# data_set = DataPreparation().load_data()
# data_set = Preprocessing().tokenization(data_set)

# train_df = data_set.sample(frac=1, random_state=1).reset_index(drop=True)
# split_index_1 = int(len(data_set) * 0.7)
# split_index_2 = int(len(data_set) * 0.8)
# train_df, test_df, validation_df = data_set[:split_index_1], data_set[split_index_1:split_index_2], data_set[split_index_2:]

# X_train = train_df['processed_tweet']
# y_train = train_df['sentiment'].to_numpy().astype(str)

# X_test = test_df['processed_tweet']
# y_test = test_df['sentiment'].to_numpy().astype(str)

# # Convert tokenized tweets to padded sequences
# desired_sequence_length = 125

# # Create a vocabulary index from tokenized data

# vocab = {word for tweet in X_train for word in tweet}

# word_to_index = {word: i + 1 for i, word in enumerate(sorted(vocab))}
# index_to_word = {i: word for word, i in word_to_index.items()}

# def encode_and_pad(data, maxlen):
#     encoded = [[word_to_index[word] for word in tweet if word in word_to_index] for tweet in data]
#     print(encoded)
#     return pad_sequences(encoded, maxlen=maxlen, padding='post', truncating='post')

# X_train_padded = encode_and_pad(X_train, desired_sequence_length)
# X_test_padded = encode_and_pad(X_test, desired_sequence_length)

# print(f"X_train_padded shape: {X_train_padded.shape}, y_train shape: {y_train.shape}")

# # Vectorization (TF-IDF Example)
# X_train_tfidf = Vectorization.vectorize_with_tfidf(train_df)
# X_test_tfidf = Vectorization.vectorize_with_tfidf(test_df)

# print(f"X_train_tfidf shape: {X_train_tfidf.shape}, y_train shape: {y_train.shape}")


#*********************************************************************************


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

from lib.data_preparation import DataPreparation
from lib.preprocessing_data import Preprocessing
from lib.vectorization import Vectorization


data_set = DataPreparation().load_data()
data_set = Preprocessing().tokenization(data_set)

# data_set = data_set.sample(frac=1, random_state=1).reset_index(drop=True, inplace=True)

split_index_1 = int(len(data_set) * 0.7)
split_index_2 = int(len(data_set) * 0.8)


train_df, test_df, validation_df = data_set[:split_index_1], data_set[split_index_1:split_index_2], data_set[split_index_2:]


X_train = train_df['tweet']
y_train = train_df['sentiment'].to_numpy().astype(str)

X_test = test_df['tweet']
y_test = test_df['sentiment'].to_numpy().astype(str)

X_validation = validation_df['tweet']
y_validation = validation_df['sentiment'].to_numpy().astype(str)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)
print(X_train_vector.shape)
print(X_test_vector.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

clf = LogisticRegression()
clf.fit(X_train_vector, y_train)
y_pred = clf.predict(X_test_vector)

print(classification_report(y_test, y_pred))
