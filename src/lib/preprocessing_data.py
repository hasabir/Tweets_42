import spacy
import pandas as pd
import nltk
from nltk.corpus import words
from nltk.stem import PorterStemmer
from nltk.corpus import words 
nltk.download('words') 



class Preprocessing:
    def __init__(self):
        pass

    @staticmethod
    def tokenization(data_set):

        data_set['processed_tweets'] = data_set['tweet'].apply(lambda row: row.split())
        return data_set


    @staticmethod
    def stemming(data_set):
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()
        tokens = data_set['tweet'].apply(lambda row: row.split())
        data_set['processed_tweets'] = tokens.apply(lambda x: [stemmer.stem(y) for y in x])
        return data_set
        
        
    @staticmethod
    def lemmatization(data_set):

        nlp = spacy.load('en_core_web_md') #python -m spacy download en_core_web_md
        data_set['processed_tweets'] = data_set['tweet'].apply(lambda x: [token.lemma_ for token in nlp(x)])
        return data_set

    
    @staticmethod
    def _get_closest_word(word, threshold=80):
        from rapidfuzz import process

        match = process.extractOne(word, words.words(), score_cutoff=threshold)
        if match and len(match[0]) > 1:
            return match[0]
        return word


    @staticmethod
    def stemming_with_misspelling(data_set):
        stemmer = PorterStemmer()
        tokens = data_set['tweet'].apply(lambda x: x.split())
        data_set['processed_tweets'] = tokens.apply(lambda x: 
                                [stemmer.stem(Preprocessing._get_closest_word(y)) for y in x])
        return data_set



    @staticmethod
    def lemmatization_with_misspelling(data_set):
        nlp = spacy.load('en_core_web_md')
        data_set['processed_tweets'] = data_set['tweet'].apply(lambda x:
                                                                [token.lemma_ for token in nlp
                                                                (" ".join(Preprocessing._get_closest_word(word) 
                                                                                for word in x.split()))
                                                                ])
        return data_set
    
    @staticmethod
    def correct_slang_words(data_set):
        from ekphrasis.classes.preprocessor import TextPreProcessor
        from ekphrasis.dicts.emoticons import emoticons
        from ekphrasis.classes.tokenizer import SocialTokenizer
        import pandas as pd

        text_processor = TextPreProcessor(
            normalize=['url', 'email', 'percent', 'money', 'phone', 'time', 'date', 'number'],
            annotate={"hashtag", "allcaps", "elongated", "repeated", "emphasis", "censored"},
            fix_html=True,
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=True,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        )

        data_set['processed_tweets'] = data_set['tweet'].apply(
            lambda x: text_processor.pre_process_doc(x) if pd.notnull(x) else x)
        return data_set









