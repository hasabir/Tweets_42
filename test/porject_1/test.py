import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('ggplot')

import nltk


df = pd.read_csv('../../Reviews.csv')

example = df['Text'][50]
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(example)

nltk.download('averaged_perceptron_tagger_eng')

tagged = nltk.pos_tag(tokens, lang="eng")


nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

entities =  nltk.chunk.ne_chunk(tagged)



from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm


sia = SentimentIntensityAnalyzer()

res= {}
for i , row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    my_id = row['Id']
    res[my_id] = sia.polarity_scores(text)
    
    
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('score')
plt.show()