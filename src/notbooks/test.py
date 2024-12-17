from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
import os
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from lib.preprocessing_data import Preprocessing
from lib.data_preparation import DataPreparation



import numpy as np

def train_word2vec(processed_tweets, embedding_size=100, window_size=3, num_negative_samples=3, learning_rate=0.1, epochs=10):
    # Create vocabulary
    vocab = set(word for tweet in processed_tweets for word in tweet)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(vocab)

    # Initialize embeddings
    main_embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_size))
    context_embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_size))

    def sigmoid(x):
        # Numerically stable sigmoid function
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def normalize_embeddings(embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms


    def get_negative_samples(vocab_size, exclude_idx, num_samples, word_freq):
        """Sample negative examples based on word frequencies."""
        probabilities = np.array([freq ** 0.75 for freq in word_freq])
        probabilities /= probabilities.sum()

        negative_samples = []
        while len(negative_samples) < num_samples:
            sampled_idx = np.random.choice(vocab_size, p=probabilities)
            if sampled_idx != exclude_idx:
                negative_samples.append(sampled_idx)
        return negative_samples


    def update_embeddings(center_idx, context_idx, label):
        # Update embeddings for one center-context pair
        center_vector = main_embeddings[center_idx]
        context_vector = context_embeddings[context_idx]

        dot_product = np.dot(center_vector, context_vector)
        prediction = sigmoid(dot_product)
        error = label - prediction

        # Gradient updates
        grad_center = error * context_vector
        grad_context = error * center_vector

        main_embeddings[center_idx] += learning_rate * grad_center
        context_embeddings[context_idx] += learning_rate * grad_context

    # Train for multiple epochs
    # Compute word frequencies for negative sampling
    from collections import Counter

    word_counts = Counter(word for tweet in processed_tweets for word in tweet)
    word_freq = np.array([word_counts[word] for word in vocab])

    # Training Loop
    for epoch in range(epochs):
        for tweet in processed_tweets:
            for center_idx, center_word in enumerate(tweet):
                center_word_idx = word_to_idx[center_word]
                start = max(center_idx - window_size, 0)
                end = min(center_idx + window_size + 1, len(tweet))

                for context_idx in range(start, end):
                    if center_idx == context_idx:
                        continue
                    context_word_idx = word_to_idx[tweet[context_idx]]

                    # Positive sample
                    update_embeddings(center_word_idx, context_word_idx, 1)

                    # Negative samples
                    negative_samples = get_negative_samples(vocab_size, center_word_idx, num_negative_samples, word_freq)
                    for negative_idx in negative_samples:
                        update_embeddings(center_word_idx, negative_idx, 0)

        # Normalize embeddings after each epoch
        main_embeddings = normalize_embeddings(main_embeddings)
        context_embeddings = normalize_embeddings(context_embeddings)
        print(f"Epoch {epoch + 1}/{epochs} completed")


    # Compute tweet embeddings as the mean of word embeddings
    tweet_embeddings = []
    for tweet in processed_tweets:
        word_indices = [word_to_idx[word] for word in tweet if word in word_to_idx]
        if word_indices:
            tweet_embedding = main_embeddings[word_indices].mean(axis=0)
        else:
            tweet_embedding = np.zeros(embedding_size)
        tweet_embeddings.append(tweet_embedding)

    return np.array(tweet_embeddings)




def train(preprocessing_method, vectorizer, model, description):
    data_set = DataPreparation().load_data()
    data_set = preprocessing_method(data_set)
    data_set = data_set.sample(frac=1, random_state=42).reset_index(drop=True)

    split_index = int(len(data_set) * 0.8)
    train_df = data_set[:split_index]
    test_df = data_set[split_index:]

    X_train = train_df['processed_tweet']
    y_train = train_df['label'].to_numpy().astype(int)
    X_test = test_df['processed_tweet']
    y_test = test_df['label'].to_numpy().astype(int)
    
    if description.split()[0] == 'word2vec':
        X_train_vector = train_word2vec(X_train)
        X_test_vector = train_word2vec(X_test)
    else:
        X_train_vector = vectorizer.fit_transform(X_train.apply(lambda row: ' '.join(row)))
        X_test_vector = vectorizer.transform(X_test.apply(lambda row: ' '.join(row)))
    
    model.fit(X_train_vector, y_train)
    y_pred = model.predict(X_test_vector)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {description}: {accuracy}")
    return description, accuracy




if __name__ == "__main__":
    preprocessing_methods = {
        # "lemmatization": Preprocessing().lemmatization,
        "tokenization": Preprocessing().tokenization,
        # "stemming": Preprocessing().stemming,
    }
    vectorizers = {
        # "TF-IDF": TfidfVectorizer(),
        # "CountVectorizer (binary=False)": CountVectorizer(binary=False),
        # "CountVectorizer (binary=True)": CountVectorizer(binary=True),
        "word2vec": PCA(n_components=2)
    }
    models = {
        "Logistic Regression": LogisticRegression(),
        # "MultinomialNB": MultinomialNB(),
        # "Linear SVC": LinearSVC()
    }

    # Dynamically run all combinations
    for preprocessing_name, preprocessing_method in preprocessing_methods.items():
        for vectorizer_name, vectorizer in vectorizers.items():
            for model_name, model in models.items():
                description = f"{vectorizer_name} with {preprocessing_name} using {model_name}"
                try:
                    train(preprocessing_method, vectorizer, model, description)
                except Exception as e:
                    print(f"Error for {description}: {e}")
                    continue








