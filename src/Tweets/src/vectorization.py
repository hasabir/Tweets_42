import pandas as pd


def	bag_of_words(data_set):
    bag_of_words = {}
    for review in data_set:
        for word in review.split():
            if word in bag_of_words:
                bag_of_words[word] += 1
            else:
                bag_of_words[word] = 1
    return pd.DataFrame.from_dict(bag_of_words, orient='index', columns=['Count']).T