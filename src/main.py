import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('ggplot')

import nltk



positive = pd.DataFrame()
# negative = pd.DataFrame()
# neutral = pd.DataFrame()

# positive["positive"] = pd.read_csv('data/processedPositive.csv', header=None ).T.squeeze()
# negative["negative"] = pd.read_csv('data/processedNegative.csv', header=None).T.squeeze()
# neutral["neutral"] = pd.read_csv('data/processedNeutral.csv', header=None).T.squeeze()



# merged = pd.merge(positive, negative, left_index=True, right_index=True)
# data_frame = pd.merge(merged, neutral, left_index=True, right_index=True)



# data_frame = data_frame.sample(frac=0.2, random_state=1)

# print(data_frame)



positif = pd.read_csv('data/processedPositive.csv')
negatif = pd.read_csv('data/processedNegative.csv')
netral = pd.read_csv('data/processedNeutral.csv')

positif_bag_of_words = {}
for review in positif:
    for word in review.split():
        if word in positif_bag_of_words:
            positif_bag_of_words[word] += 1
        else:
            positif_bag_of_words[word] = 1

negative_bag_of_words = {}
for review in negatif:
    for word in review.split():
        if word in negative_bag_of_words:
            negative_bag_of_words[word] += 1
        else:
            negative_bag_of_words[word] = 1

netral_bag_of_words = {}
for review in netral:
    for word in review.split():
        if word in netral_bag_of_words:
            netral_bag_of_words[word] += 1
        else:
            netral_bag_of_words[word] = 1

positif_bag_of_words = pd.DataFrame.from_dict(positif_bag_of_words, orient='index', columns=['Positif']).T
negative_bag_of_words = pd.DataFrame.from_dict(negative_bag_of_words, orient='index', columns=['Negatif']).T
netral_bag_of_words = pd.DataFrame.from_dict(netral_bag_of_words, orient='index', columns=['Netral']).T


# bag_of_words = pd.merge(positif_bag_of_words, negative_bag_of_words, left_index=True, right_index=True)
# bag_of_words = pd.merge(bag_of_words, netral_bag_of_words, left_index=True, right_index=True)

# print(bag_of_words)

print(positif_bag_of_words)
print(negative_bag_of_words)
print(netral_bag_of_words)

print('-----------------------------')


test = pd.concat([positif_bag_of_words, negative_bag_of_words, netral_bag_of_words], axis=0)
tesr = test.fillna(0, inplace=True)
print(test)