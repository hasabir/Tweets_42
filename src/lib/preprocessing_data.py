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
            stemmed_data[column] = tokenized_data_frame[column].apply(
                lambda row: [stemmer.stem(word) for word in row])
        

        return stemmed_data

    @staticmethod
    def lemmatization(data_set) -> list:

        nlp = spacy.load('en_core_web_md') #python -m spacy download en_core_web_md
        lemmatized_words = pd.DataFrame()
        for column in data_set.columns:
            lemmatized_words[column] = data_set[column].astype(str).apply(
                lambda row: [token.lemma_ for token in nlp(row)]
            )
        return lemmatized_words

    
    @staticmethod
    def _get_closest_word(word, threshold=80):
        from rapidfuzz import process

        match = process.extractOne(word, words.words(), score_cutoff=threshold)
        if match and len(match[0]) > 1:
            return match[0]
        return word


    @staticmethod
    def stemming_with_misspelling(data_frame) -> list:
        stemmer = PorterStemmer()
        corrected_stemmed_data = pd.DataFrame()
        tokenized_data_frame = Preprocessing.tokenization(data_frame)
        for column in tokenized_data_frame.columns:
            corrected_stemmed_data[column] = tokenized_data_frame[column].apply(
                lambda row: [stemmer.stem( Preprocessing._get_closest_word(token)) for token in row])
        return corrected_stemmed_data



    @staticmethod
    def lemmatization_with_misspelling(data_set):
        nlp = spacy.load('en_core_web_md')
        corrected_lemmatizide_data = pd.DataFrame()

        for column in data_set.columns:
            corrected_lemmatizide_data[column] = data_set[column].astype(str).apply(
                lambda row: [token.lemma_ 
                             for token in nlp(" ".join(Preprocessing._get_closest_word(token) 
                                                       for token in row.split()))]
            )
        
        return corrected_lemmatizide_data
