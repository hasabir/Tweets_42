import spacy
import pandas as pd
import nltk
from nltk.corpus import words
from nltk.metrics.distance import edit_distance
from nltk.stem import PorterStemmer
from nltk.metrics.distance import jaccard_distance 
from nltk.util import ngrams
from nltk.corpus import words 
from nltk.metrics.distance  import edit_distance 
nltk.download('words') 



class Preprocessing:
    def __init__(self):
        pass

    @staticmethod
    def tokenization(data_set) -> list:
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        tokens = pd.DataFrame()
        for column in data_set.columns:
            tokens[column] = data_set[column].astype(str).apply(tokenizer.tokenize)
        return tokens


    @staticmethod
    def stemming(data_frame) -> list:
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()
        stemmed_data = pd.DataFrame()
        tokenized_data_frame = Preprocessing.tokenization(data_frame)
        
        for column in tokenized_data_frame.columns:
            stemmed_data[column] = tokenized_data_frame[column].apply(lambda row: [stemmer.stem(word) for word in row])
        

        return stemmed_data

    @staticmethod
    def lemmatization(data_frame) -> list:
        import spacy
        
        #python -m spacy download en_core_web_md

        nlp = spacy.load('en_core_web_md')
        lemmatized_words = []
        for column in data_frame:
            data = nlp(data)
            lemmatized_words.append([token.lemma_ for token in data])
        return lemmatized_words

    # @staticmethod
    # def n_grams(data_frame, n=2):
    #     n_grams = []
    #     for data in data_frame:
    #         n_grams.append(list(ngrams(data, n)))
    #     return n_grams


    # @staticmethod
    # def stemming_with_jaccard_distance(data_frame) -> list:
    #     correct_words = words.words()
    #     stemmer = PorterStemmer()
    #     stemmed_words = []
    #     tokenized_data_frame = Preprocessing.tokenization(data_frame)

    #     for data in tokenized_data_frame:
    #         corrected_data = []
    #         for word in data:
    #             distances = []
    #             word_bigrams = set(ngrams(word, 2))
    #             if word_bigrams:
    #                 distances = [
    #                     (jaccard_distance(word_bigrams, set(ngrams(w, 2))), w)
    #                     for w in correct_words
    #                     if set(ngrams(w, 2))
    #                 ]
    #             closest_word = min(distances, key=lambda x: x[0])[1] if distances else word
    #             stemmed_word = stemmer.stem(closest_word)
    #             corrected_data.append(stemmed_word)
    #         stemmed_words.append(corrected_data)
    #     return stemmed_words

    # @staticmethod
    # def stemming_with_levenshtein_distance(data_frame) -> list:
    #     correct_words = words.words()
    #     stemmer = PorterStemmer()
    #     stemmed_words = []
    #     tokenized_data_frame = Preprocessing.tokenization(data_frame)
    #     for data in tokenized_data_frame:
    #         corrected_data = []
    #         for word in data:
    #             temp = [(edit_distance(word, w),w) for w in correct_words if w[0]==word[0]] 
    #             closest_word = min(temp, key=lambda x: x[0])[1] if temp else word
    #             stemmed_word = stemmer.stem(closest_word)
    #             corrected_data.append(stemmed_word)
    #         stemmed_words.append(corrected_data)
    #     return stemmed_words


    # @staticmethod
    # def lemmatization_with_misspelling(data_frame):
    #     nlp = spacy.load('en_core_web_md')
    #     lemmatized_words = []
    #     correct_words = words.words()
        
    #     for data in data_frame:
    #         data = nlp(data)
    #         corrected_data = []
    #         for word in data:
    #             word_text = word.text
    #             temp = [(edit_distance(word_text, w), w) for w in correct_words if w[0] == word_text[0]]
    #             closest_word = min(temp, key=lambda x: x[0])[1] if temp else word_text
    #             lemma = nlp(closest_word)[0].lemma_
    #             corrected_data.append(lemma)
            
    #         lemmatized_words.append(corrected_data)
        
    #     return lemmatized_words
