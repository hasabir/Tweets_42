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


    @staticmethod
    def remove_stopwords_from_text(text, stop_words):
        if not isinstance(text, str):
            return ''
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    @staticmethod
    def remove_stopwords(data_frame: pd.DataFrame) -> pd.DataFrame:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english')) #python -m nltk.downloader stopwords
        stop_words = set(stopwords.words('english'))
        data_frame['tweet'] = data_frame['tweet'].apply( Preprocessing.remove_stopwords_from_text, args=(stop_words,))
        return data_frame

    @staticmethod
    def lemmatization_with_stopwords_removal(data_frame) -> pd.DataFrame:
        data_frame = Preprocessing.remove_stopwords(data_frame)
        # return data_frame
        return Preprocessing.lemmatization(data_frame)
    
    @staticmethod
    def remove_duplicates(tweet_vectors):
        from sklearn.metrics.pairwise import cosine_similarity
        tweet_vectors.reset_index(drop=True, inplace=True)
        similarity_matrix = cosine_similarity(tweet_vectors)
        similarity_df = pd.DataFrame(similarity_matrix)
        
        tweet_similarity_scores = similarity_df.sum(axis=1) - 1  
        
        top_10_similar_tweets = tweet_similarity_scores.nlargest(10)

        tweet_vectors = tweet_vectors.drop(index=top_10_similar_tweets.index[1:])
        return tweet_vectors



data_set = pd.read_csv('../data/raw_splits/train.csv')
data_set = Preprocessing.lemmatization_with_stopwords_removal(data_set)
# print(data_set.head())

