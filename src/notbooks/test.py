
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

from lib.data_preparation import DataPreparation
from lib.preprocessing_data import Preprocessing


data_set = DataPreparation().load_data()
data_set = Preprocessing().tokenization(data_set)

data_set = data_set.sample(frac=0.005).reset_index(drop=True)


import numpy as np

WINDOW_SIZE = 5
NUM_NEGATIVE_SAMPLES = 5

data = []

for processed_tweet in data_set['processed_tweet']:
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



import pandas as pd

df = pd.DataFrame(columns=['center_word', 'context_word', 'label'], data=data)
words = np.intersect1d(df.context_word, df.center_word)
df = df[(df.center_word.isin(words)) & (df.context_word.isin(words))].reset_index(drop=True)

def sigmoid(v, scale=1):
    return 1 / (1 + np.exp(-scale*v))

def normalize_data(data):
    row_norms = np.sqrt((data.values**2).sum(axis=1)).reshape(-1,1)
    return data.divide(row_norms, axis='index')


def update_embeddings(df, main_embeddings, context_embeddings, learning_rate, debug=False):
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


EMBEDDING_SIZE = 5

main_embeddings = np.random.normal(0,0.1,(len(words), EMBEDDING_SIZE))
row_norms = np.sqrt((main_embeddings**2).sum(axis=1)).reshape(-1,1)
main_embeddings = main_embeddings / row_norms

context_embeddings = np.random.normal(0,0.1,(len(words), EMBEDDING_SIZE))
row_norms = np.sqrt((context_embeddings**2).sum(axis=1)).reshape(-1,1)
context_embeddings = context_embeddings / row_norms

main_embeddings = pd.DataFrame(data=main_embeddings, index=words)
context_embeddings = pd.DataFrame(data=context_embeddings, index=words)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
transf_embeddings = pca.fit_transform(main_embeddings.values)
words_used = main_embeddings.index











