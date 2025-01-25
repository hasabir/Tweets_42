import sys
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from lib.preprocessing_data import Preprocessing
from lib.data_preparation import DataPreparation


# Word2VecTorch Implementation
class Word2VecTorch(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Word2VecTorch, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_size)

    def forward(self, center_word, context_word):
        center_embed = self.embeddings(center_word)
        context_embed = self.context_embeddings(context_word)
        scores = torch.matmul(center_embed, context_embed.t())
        return scores


def train_torch_word2vec(processed_tweetss, embedding_size=25, window_size=3, epochs=10, learning_rate=0.01):
    vocab = set(word for tweet in processed_tweetss for word in tweet)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    vocab_size = len(vocab)
    model = Word2VecTorch(vocab_size, embedding_size).cuda()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for tweet in processed_tweetss:
            for center_idx, center_word in enumerate(tweet):
                center_word_idx = torch.tensor(word_to_idx[center_word]).cuda()
                start = max(center_idx - window_size, 0)
                end = min(center_idx + window_size + 1, len(tweet))

                for context_idx in range(start, end):
                    if center_idx == context_idx:
                        continue
                    context_word_idx = torch.tensor(word_to_idx[tweet[context_idx]]).cuda()

                    optimizer.zero_grad()
                    output = model(center_word_idx.unsqueeze(0), context_word_idx.unsqueeze(0))
                    loss = criterion(output, torch.tensor([1.0]).cuda())
                    loss.backward()
                    optimizer.step()

        print(f"Epoch {epoch + 1} completed")

    return model


# Load Pretrained Word2Vec Embeddings
def load_pretrained_word2vec(processed_tweetss, embedding_size=50, path_to_embeddings='path/to/glove.6B.50d.txt'):
    word_vectors = KeyedVectors.load_word2vec_format(path_to_embeddings, binary=False)
    tweet_embeddings = []
    for tweet in processed_tweetss:
        embeddings = [word_vectors[word] for word in tweet if word in word_vectors]
        if embeddings:
            tweet_embeddings.append(np.mean(embeddings, axis=0))
        else:
            tweet_embeddings.append(np.zeros(embedding_size))  # Zero vector for tweets without known words
    return np.array(tweet_embeddings)


# Training Pipeline
def train(preprocessing_method, vectorizer, model, description):
    data_set = DataPreparation().load_data()
    data_set = preprocessing_method(data_set)

    train_df, test_df = train_test_split(
        data_set, test_size=0.2, random_state=1, stratify=data_set['label']
    )

    X_train = train_df['processed_tweets']
    y_train = train_df['label'].to_numpy().astype(int)
    X_test = test_df['processed_tweets']
    y_test = test_df['label'].to_numpy().astype(int)

    if description.startswith("word2vec"):
        X_train_vector = load_pretrained_word2vec(X_train)
        X_test_vector = load_pretrained_word2vec(X_test)
    else:
        X_train_vector = vectorizer.fit_transform(X_train.apply(lambda row: ' '.join(row)))
        X_test_vector = vectorizer.transform(X_test.apply(lambda row: ' '.join(row)))

    model.fit(X_train_vector, y_train)
    y_pred = model.predict(X_test_vector)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {description}: {accuracy}")
    return accuracy


# Configurations
preprocessing_methods = {
    "lemmatization": Preprocessing().lemmatization,
    "tokenization": Preprocessing().tokenization,
    "stemming": Preprocessing().stemming,
}

vectorizers = {
    # "TF-IDF": TfidfVectorizer(),
    # "Bag of words": CountVectorizer(binary=False),
    # "o or 1 if word exists": CountVectorizer(binary=True),
    "word2vec": None,
}

models = {
    "Logistic Regression": LogisticRegression(),
    "MultinomialNB": MultinomialNB(),
    "SVC": SVC(),
}

# Run Experiments
results = []

for preprocessing_name, preprocessing_method in preprocessing_methods.items():
    for vectorizer_name, vectorizer in vectorizers.items():
        for model_name, model in models.items():
            description = f"{vectorizer_name} + {preprocessing_name} + {model_name}"
            accuracy = train(preprocessing_method, vectorizer, model, description)
            results.append((description, accuracy))
