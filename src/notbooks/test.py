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
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from lib.preprocessing_data import Preprocessing
from lib.data_preparation import DataPreparation


def train_word2vec(processed_tweets):
    WINDOW_SIZE = 5
    NUM_NEGATIVE_SAMPLES = 5

    data = []

    for processed_tweet in processed_tweets:
        for idx, center_word in enumerate(processed_tweet):
            start = max(idx - WINDOW_SIZE, 0)
            end = min(idx + WINDOW_SIZE + 1, len(processed_tweet))
            context_words = [processed_tweet[i] for i in range(start, end) if i != idx]
            for context_word in context_words:
                data.append([center_word, context_word, 1])
            negative_candidates = [
                word for word in processed_tweet if word != center_word and word not in context_words
            ]
            negative_samples = np.random.choice(negative_candidates, 
                                                min(NUM_NEGATIVE_SAMPLES, len(negative_candidates)), 
                                                replace=False)
            for negative_sample in negative_samples:
                data.append([center_word, negative_sample, 0])




    df = pd.DataFrame(columns=['center_word', 'context_word', 'label'], data=data)
    words = np.intersect1d(df.context_word, df.center_word)
    df = df[(df.center_word.isin(words)) & (df.context_word.isin(words))].reset_index(drop=True)

    def sigmoid(v, scale=1):
        return 1 / (1 + np.exp(-scale*v))

    def normalize_data(data):
        row_norms = np.sqrt((data.values**2).sum(axis=1)).reshape(-1,1)
        return data.divide(row_norms, axis='index')


    def update_embeddings(df, main_embeddings, context_embeddings, learning_rate):
        main_embeddings_center = main_embeddings.loc[df.center_word].values
        context_embeddings_context = context_embeddings.loc[df.context_word].values
        diffs = context_embeddings_context - main_embeddings_center
        
        dot_prods = np.sum(main_embeddings_center * context_embeddings_context, axis=1)
        scores = sigmoid(dot_prods)
        errors = (df.label - scores).values.reshape(-1,1)
        
        updates = diffs*errors*learning_rate
        updates_df = pd.DataFrame(data=updates)
        updates_df['center_word'] = df.center_word
        updates_df['context_word'] = df.context_word
        updates_df_center = updates_df.groupby('center_word').sum()
        updates_df_context = updates_df.groupby('context_word').sum()

        main_embeddings += updates_df_center.loc[main_embeddings.index]
        context_embeddings -= updates_df_context.loc[context_embeddings.index]
        
        main_embeddings = normalize_data(main_embeddings)
        context_embeddings = normalize_data(context_embeddings)
        
        return main_embeddings, context_embeddings


    EMBEDDING_SIZE = 5144
    

    
    main_embeddings = np.random.normal(0, 0.1, (len(words), EMBEDDING_SIZE))
    # main_embeddings = np.random.normal(0, 0.1, (len(words), EMBEDDING_SIZE)).astype(np.float32)
    row_norms = np.sqrt((main_embeddings**2).sum(axis=1)).reshape(-1,1)
    main_embeddings = main_embeddings / row_norms

    # context_embeddings = np.random.normal(0, 0.1, (len(words), EMBEDDING_SIZE)).astype(np.float32)
    context_embeddings = np.random.normal(0,0.1,(len(words), EMBEDDING_SIZE))
    row_norms = np.sqrt((context_embeddings**2).sum(axis=1)).reshape(-1,1)
    context_embeddings = context_embeddings / row_norms

    main_embeddings = pd.DataFrame(data=main_embeddings, index=words)
    context_embeddings = pd.DataFrame(data=context_embeddings, index=words)
    
    # def update_embeddings_in_batches(df, main_embeddings, context_embeddings, learning_rate, batch_size=1000):
    #     for i in range(0, len(df), batch_size):
    #         batch = df.iloc[i:i + batch_size]
    #         main_embeddings, context_embeddings = update_embeddings(batch, main_embeddings, context_embeddings, learning_rate)
    #     return main_embeddings, context_embeddings

    
    learning_rate = 0.1
    for _ in range(10):
        main_embeddings, context_embeddings = update_embeddings(df, main_embeddings, context_embeddings, learning_rate)
        # main_embeddings, context_embeddings = update_embeddings_in_batches(df, main_embeddings, context_embeddings, learning_rate)

    tweet_embeddings = []
    for tweet in processed_tweets:
        tweet_words = [word for word in tweet if word in main_embeddings.index]
        if tweet_words:
            tweet_embedding = main_embeddings.loc[tweet_words].mean(axis=0)
        else:
            tweet_embedding = np.zeros(main_embeddings.shape[1])
        tweet_embeddings.append(tweet_embedding)
    
    return np.array(tweet_embeddings)

    

from sklearn.decomposition import PCA


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
        print(f"----------------- for {description.split()[0]}----------------------")
        # print(X_train_vector)
    else:
        X_train_vector = vectorizer.fit_transform(X_train.apply(lambda row: ' '.join(row)))
        print(f"----------------- for {description.split()[0]}----------------------")
        # print(X_train_vector)
        X_test_vector = vectorizer.transform(X_test.apply(lambda row: ' '.join(row)))
    
    print(f"y_train: {y_train.shape} | X_train_vector: {X_train_vector.shape}")
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








