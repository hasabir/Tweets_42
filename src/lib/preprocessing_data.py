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
    def tokenization(data_set) -> list:
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

        data_set['processed_tweet'] = data_set['tweet'].apply(lambda x: tokenizer.tokenize(x))
        return data_set


    @staticmethod
    def stemming(data_set) -> list:
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()
        tokens = data_set['tweet'].apply(lambda x: x.split())
        data_set['processed_tweet'] = tokens.apply(lambda x: [stemmer.stem(y) for y in x])
        return data_set
        
        
    @staticmethod
    def lemmatization(data_set) -> list:

        nlp = spacy.load('en_core_web_md') #python -m spacy download en_core_web_md
        data_set['processed_tweet'] = data_set['tweet'].apply(lambda x: [token.lemma_ for token in nlp(x)])
        return data_set

    
    @staticmethod
    def _get_closest_word(word, threshold=80):
        from rapidfuzz import process

        match = process.extractOne(word, words.words(), score_cutoff=threshold)
        if match and len(match[0]) > 1:
            return match[0]
        return word


    @staticmethod
    def stemming_with_misspelling(data_set) -> list:
        stemmer = PorterStemmer()
        tokens = data_set['tweet'].apply(lambda x: x.split())
        data_set['processed_tweet'] = tokens.apply(lambda x: 
                                [stemmer.stem(Preprocessing._get_closest_word(y)) for y in x])
        return data_set



    @staticmethod
    def lemmatization_with_misspelling(data_set):
        nlp = spacy.load('en_core_web_md')
        data_set['processed_tweet'] = data_set['tweet'].apply(lambda x:
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

        data_set['processed_tweet'] = data_set['tweet'].apply(
            lambda x: " ".join(text_processor.pre_process_doc(x) if pd.notnull(x) else x))
        return data_set











# import spacy
# import pandas as pd
# import nltk
# from nltk.corpus import words
# from nltk.stem import PorterStemmer
# from nltk.corpus import words 
# nltk.download('words') 



# class Preprocessing:
#     def __init__(self):
#         pass

#     @staticmethod
#     def tokenization(data_set) -> list:
#         tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
#         tokens = pd.DataFrame()
#         for column in data_set.columns:
#             tokens[column] = data_set[column].astype(str).apply(tokenizer.tokenize)
#         return tokens


#     @staticmethod
#     def stemming(data_frame) -> list:
#         from nltk.stem import PorterStemmer

#         stemmer = PorterStemmer()
#         stemmed_data = pd.DataFrame()
#         tokenized_data_frame = Preprocessing.tokenization(data_frame)
        
#         for column in tokenized_data_frame.columns:
#             stemmed_data[column] = tokenized_data_frame[column].apply(
#                 lambda row: [stemmer.stem(word) for word in row])
        

#         return stemmed_data

#     @staticmethod
#     def lemmatization(data_set) -> list:

#         nlp = spacy.load('en_core_web_md') #python -m spacy download en_core_web_md
#         lemmatized_words = pd.DataFrame()
#         for column in data_set.columns:
#             lemmatized_words[column] = data_set[column].astype(str).apply(
#                 lambda row: [token.lemma_ for token in nlp(row)]
#             )
#         return lemmatized_words

    
#     @staticmethod
#     def _get_closest_word(word, threshold=80):
#         from rapidfuzz import process

#         match = process.extractOne(word, words.words(), score_cutoff=threshold)
#         if match and len(match[0]) > 1:
#             return match[0]
#         return word


#     @staticmethod
#     def stemming_with_misspelling(data_frame) -> list:
#         stemmer = PorterStemmer()
#         corrected_stemmed_data = pd.DataFrame()
#         tokenized_data_frame = Preprocessing.tokenization(data_frame)
#         for column in tokenized_data_frame.columns:
#             corrected_stemmed_data[column] = tokenized_data_frame[column].apply(
#                 lambda row: [stemmer.stem( Preprocessing._get_closest_word(token)) for token in row])
#         return corrected_stemmed_data



#     @staticmethod
#     def lemmatization_with_misspelling(data_set):
#         nlp = spacy.load('en_core_web_md')
#         corrected_lemmatizide_data = pd.DataFrame()

#         for column in data_set.columns:
#             corrected_lemmatizide_data[column] = data_set[column].astype(str).apply(
#                 lambda row: [token.lemma_ 
#                              for token in nlp(" ".join(Preprocessing._get_closest_word(token) 
#                                                        for token in row.split()))]
#             )
        
#         return corrected_lemmatizide_data
    
#     @staticmethod
#     def correct_slang_words(data_set):
#         from ekphrasis.classes.preprocessor import TextPreProcessor
#         from ekphrasis.dicts.emoticons import emoticons
#         from ekphrasis.classes.tokenizer import SocialTokenizer
#         import pandas as pd

#         text_processor = TextPreProcessor(
#             normalize=['url', 'email', 'percent', 'money', 'phone', 'time', 'date', 'number'],
#             annotate={"hashtag", "allcaps", "elongated", "repeated", "emphasis", "censored"},
#             fix_html=True,
#             unpack_hashtags=True,
#             unpack_contractions=True,
#             spell_correct_elong=True,
#             tokenizer=SocialTokenizer(lowercase=True).tokenize,
#             dicts=[emoticons]
#         )

#         processed_data = pd.DataFrame()
#         for column in data_set.columns:
#             processed_data[column] = data_set[column].astype(str).apply(
#                 lambda row: " ".join(text_processor.pre_process_doc(row)) if pd.notnull(row) else row
#             )
#         return processed_data
