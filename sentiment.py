import pandas as pd
import numpy as np
import contractions
import re
import string
import seaborn as sn
import random
from matplotlib import pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def replace_retweet(tweet, default=""):
    tweet = re.sub('RT\s+', default, tweet)
    return tweet


def replace_user(tweet, default="twitteruser"):
    tweet = re.sub('\B@\w+', default, tweet)
    return tweet


# def replace_emoji(tweet):
#     tweet = emoji.demojize(tweet)
#     return tweet


def replace_url(tweet, default=""):
    tweet = re.sub('(http|https):\/\/\S+', default, tweet)
    return tweet


def replace_hashtag(tweet, default=""):
    tweet = re.sub('#+', default, tweet)
    return tweet


def to_lowercase(tweet):
    tweet = tweet.lower()
    return tweet


def word_repetition(tweet):
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
    return tweet


def punct_repetition(tweet, default=""):
    tweet = re.sub(r'[\?\.\!]+(?=[\?\.\!])', default, tweet)
    return tweet


def fix_contractions(tweet):
    for k, v in contractions.contractions_dict.items():
        tweet = tweet.replace(k, v)
    return tweet


stop_words = set(stopwords.words('english'))


def custom_tokenize(tweet,
                    keep_punct=False,
                    keep_alnum=False,
                    keep_stop=False):
    token_list = word_tokenize(tweet)

    if not keep_punct:
        token_list = [token for token in token_list
                      if token not in string.punctuation]

    if not keep_alnum:
        token_list = [token for token in token_list if token.isalpha()]

    if not keep_stop:
        stop_words = set(stopwords.words('english'))
        stop_words.discard('not')
        token_list = [token for token in token_list if not token in stop_words]

    return token_list


tokens = ["manager", "management", "managing"]

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snoball_stemmer = SnowballStemmer('english')


def stem_tokens(tokens, stemmer):
    token_list = []
    for token in tokens:
        token_list.append(stemmer.stem(token))
    return token_list


def process_tweet(tweet, verbose=False):
    if verbose: print("Initial tweet: {}".format(tweet))

    ## Twitter Features
    tweet = replace_retweet(tweet)  # replace retweet
    tweet = replace_user(tweet, "")  # replace user tag
    tweet = replace_url(tweet)  # replace url
    tweet = replace_hashtag(tweet)  # replace hashtag
    if verbose: print("Post Twitter processing tweet: {}".format(tweet))

    ## Word Features
    tweet = to_lowercase(tweet)  # lower case
    tweet = fix_contractions(tweet)  # replace contractions
    tweet = punct_repetition(tweet)  # replace punctuation repetition
    tweet = word_repetition(tweet)  # replace word repetition

    if verbose: print("Post Word processing tweet: {}".format(tweet))

    ## Tokenization & Stemming
    tokens = custom_tokenize(tweet, keep_alnum=False, keep_stop=False)  # tokenize
    stemmer = SnowballStemmer("english")  # define stemmer
    stem = stem_tokens(tokens, stemmer)  # stem tokens

    return stem


df = pd.read_csv("tweet_data.csv")
df["tokens"] = df["tweet_text"].apply(process_tweet)
df["tweet_sentiment"] = df["sentiment"].apply(lambda i: 1
if i == "positive" else 0)
df.head(10)
X = df["tokens"].tolist()
y = df["tweet_sentiment"].tolist()


def build_freqs(tweet_list, sentiment_list):
    freqs = {}
    for tweet, sentiment in zip(tweet_list, sentiment_list):
        for word in tweet:
            pair = (word, sentiment)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs


def tweet_to_freq(tweet, freqs):
    x = np.zeros((2,))
    for word in tweet:
        if (word, 1) in freqs:
            x[0] += freqs[(word, 1)]
        if (word, 0) in freqs:
            x[1] += freqs[(word, 0)]
    return x


def fit_cv(tweet_corpus):
    cv_vect = CountVectorizer(tokenizer=lambda x: x,
                              preprocessor=lambda x: x)
    cv_vect.fit(tweet_corpus)
    return cv_vect


def fit_tfidf(tweet_corpus):
    tf_vect = TfidfVectorizer(preprocessor=lambda x: x,
                              tokenizer=lambda x: x)
    tf_vect.fit(tweet_corpus)
    return tf_vect


def plot_confusion(cm):
    plt.figure(figsize=(5, 5))
    sn.heatmap(cm, annot=True, cmap="Blues", fmt='.0f')
    plt.xlabel("Prediction")
    plt.ylabel("True value")
    plt.title("Confusion Matrix")
    return sn


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0,
                                                    train_size=0.80)


# print("Size of X_train: {}".format(len(X_train)))
# print("Size of y_train: {}".format(len(y_train)))
# print("\n")
# print("Size of X_test: {}".format(len(X_test)))
# print("Size of y_test: {}".format(len(y_test)))
# print("\n")
# print("Train proportion: {:.0%}".format(len(X_train)/
#                                         (len(X_train)+len(X_test))))
#
#
# id = random.randint(0,len(X_train))
# print("Train tweet: {}".format(X_train[id]))
# print("Sentiment: {}".format(y_train[id]))

# linear regression

def fit_lr(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


freqs = build_freqs(X_train, y_train)
X_train_pn = [tweet_to_freq(tweet, freqs) for tweet in X_train]
X_test_pn = [tweet_to_freq(tweet, freqs) for tweet in X_test]

model_lr_pn = fit_lr(X_train_pn, y_train)

# count vector


cv = fit_cv(X_train)
X_train_cv = cv.transform(X_train)
X_test_cv = cv.transform(X_test)

model_lr_cv = fit_lr(X_train_cv, y_train)

# tf idf
tf = fit_tfidf(X_train)
X_train_tf = tf.transform(X_train)
X_test_tf = tf.transform(X_test)
model_lr_tf = fit_lr(X_train_tf, y_train)

y_pred_lr_pn = model_lr_pn.predict(X_test_pn)
print("LR Model Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred_lr_pn)))
# plot_confusion(confusion_matrix(y_test, y_pred_lr_pn))

y_pred_lr_cv = model_lr_cv.predict(X_test_cv)
print("CV Model Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred_lr_cv)))
# plot_confusion(confusion_matrix(y_test, y_pred_lr_cv))

y_pred_lr_tf = model_lr_tf.predict(X_test_tf)
print("TF-IDF Model Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred_lr_tf)))
# plot_confusion(confusion_matrix(y_test, y_pred_lr_tf))


def predict_tweet(tweet):
    processed_tweet = process_tweet(tweet)
    transformed_tweet = tf.transform([processed_tweet])
    prediction = model_lr_tf.predict(transformed_tweet)

    if prediction == 1:
        return "Prediction is positive sentiment"
    else:
        return "Prediction is negative sentiment"


print("\n")
for i in range(5):
    tweet_id = random.randint(0, len(df))
    tweet = df.iloc[tweet_id]["tweet_text"]
    print(tweet)
    print(predict_tweet(tweet))
    print("\n")




# intro about sentiment analysis

# explaining the methods used

# evaluation matrix

# aiming to improve accuracy

# extra

# LSTM
