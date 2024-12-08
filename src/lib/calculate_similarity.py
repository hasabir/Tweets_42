import pandas as pd

# Original data frame
data = {
    'positive': [
        "thank kpop crying joy",
        "share lovethanks top new followers week happy",
        "go back school music think time happy",
        "bus bangsar heading awan besar good day happy",
        "well fun thanks stopping giving shit tonight fun",
        "thanks recent follow much appreciated happy way"
    ],
    'negative': [
        "rude unhappy",
        "im lonely unhappy",
        "major waffle cravings right sad",
        "talking girl behind jin unhappy",
        "true khilado kuch unhappy",
        "shameful"
    ],
    'neutral': [
        "kids orphaned parents branded hacked death",
        "akhilesh keeps options open",
        "bjp leaders dock",
        "modi tested waters letter",
        "more also epaper",
        "gave mp bus board flightas goodwill"
    ]
}

df = pd.DataFrame(data)

# Reshape to long-form structure
long_df = df.melt(var_name="sentiment", value_name="tweet")
long_df["tweet_id"] = range(1, len(long_df) + 1)  # Assign unique IDs
long_df.set_index("tweet_id", inplace=True)

print(long_df.head())


import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_tweet(tweet):
    tokens = tweet.split()
    tokens = [word for word in tokens if word.lower() not in stop_words]  # Remove stop words
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)


long_df['cleaned_tweet'] = long_df['tweet'].apply(preprocess_tweet)
vectorizer = TfidfVectorizer()
tweet_vectors = vectorizer.fit_transform(long_df['cleaned_tweet'])


from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity
similarity_matrix = cosine_similarity(tweet_vectors)

# Create a similarity DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=long_df.index, columns=long_df.index)

# Get top-10 similar tweets for each tweet
top_10_similar_tweets = similarity_df.apply(lambda row: row.nlargest(11).iloc[1:], axis=1)  # Exclude self
print(top_10_similar_tweets)

print(long_df.loc[top_10_similar_tweets.index, 'tweet'])
print(long_df.loc[top_10_similar_tweets.index])
