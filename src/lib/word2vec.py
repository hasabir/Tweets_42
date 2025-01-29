import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

from collections import Counter
import numpy as np


import numpy as np
from collections import Counter

class Word2Vec:
    def __init__(self, processed_tweetss, embedding_size, window_size=3, num_negative_samples=3, learning_rate=0.1, epochs=50):
        self.processed_tweetss = processed_tweetss
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.num_negative_samples = num_negative_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.vocab = set(word for tweet in processed_tweetss for word in tweet)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.vocab)
        
        scale = np.sqrt(2 / (self.vocab_size + self.embedding_size))  # Xavier/Glorot initialization
        self.main_embeddings = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_size))
        self.context_embeddings = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_size))
        
        self.word_counts = Counter(word for tweet in processed_tweetss for word in tweet)
        self.word_freq = np.array([self.word_counts[word] for word in self.vocab])

    def _sigmoid(self, x):
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def _normalize_embeddings(self):
        norms = np.linalg.norm(self.main_embeddings, axis=1, keepdims=True)
        self.main_embeddings /= norms
        norms = np.linalg.norm(self.context_embeddings, axis=1, keepdims=True)
        self.context_embeddings /= norms

    def _get_negative_samples(self, exclude_idx):
        probabilities = np.array([freq ** 0.75 for freq in self.word_freq])
        probabilities /= probabilities.sum()
        negative_samples = set()

        while len(negative_samples) < self.num_negative_samples:
            sampled_idx = np.random.choice(self.vocab_size, p=probabilities)
            if sampled_idx != exclude_idx:
                negative_samples.add(sampled_idx)
        return list(negative_samples)

    def _update_embeddings(self, center_idx, context_idx, label):
        center_vector = self.main_embeddings[center_idx]
        context_vector = self.context_embeddings[context_idx]

        dot_product = np.dot(center_vector, context_vector)
        prediction = self._sigmoid(dot_product)
        error = label - prediction

        grad_center = error * context_vector
        grad_context = error * center_vector

        self.main_embeddings[center_idx] += self.learning_rate * grad_center
        self.context_embeddings[context_idx] += self.learning_rate * grad_context

    def _train(self):
        for epoch in range(self.epochs):
            for tweet in self.processed_tweetss:
                for center_idx, center_word in enumerate(tweet):
                    center_word_idx = self.word_to_idx[center_word]
                    start = max(center_idx - self.window_size, 0)
                    end = min(center_idx + self.window_size + 1, len(tweet))

                    for context_idx in range(start, end):
                        if center_idx == context_idx:
                            continue
                        context_word_idx = self.word_to_idx[tweet[context_idx]]

                        self._update_embeddings(center_word_idx, context_word_idx, 1)

                        negative_samples = self._get_negative_samples(center_word_idx)
                        for negative_idx in negative_samples:
                            self._update_embeddings(center_word_idx, negative_idx, 0)
            
            self._normalize_embeddings()

    def word2vec(self):
        self._train()
        tweet_embeddings = []
        for tweet in self.processed_tweetss:
            word_indices = [self.word_to_idx[word] for word in tweet if word in self.word_to_idx]
            if word_indices:
                tweet_embedding = self.main_embeddings[word_indices].mean(axis=0)
            else:
                tweet_embedding = np.zeros(self.embedding_size)
            tweet_embeddings.append(tweet_embedding)

        return np.array(tweet_embeddings)
