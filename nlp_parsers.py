import re
import json
import nltk
import requests
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_stopwords(link_to_stpwds = "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d\
                                    /raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt"):
    """
    Gets a comprehensive list of stopwords from the web and adds these words together with the stopwords offered by the
    NLTK library

    :param link_to_stpwds: the link to the site where the comprehensive stopwords are
    :return: a full list of stopwords
    """

    # Load in a list of stopwords from the web
    stopwords_list = requests.get(link_to_stpwds).content
    stopwds = list(set(stopwords_list.decode().splitlines()))

    # Load in the list of stopwords offered by NLTK
    nltk_stopwords = stopwords.words('english')

    # Add the two lists of stopwords together
    stopwds += nltk_stopwords

    return stopwds


def get_ngram_freq(n_gram, words):
    """
    Finds the n_gram frequencies of the text and returns data as a list of tuples

    :param n_gram: value that decides number of contiguous items to get frequency of
    :param words: the list of words from the text, excluding stopwords
    :return: a list of tuples (n-gram, frequency) sorted by frequency
    """
    # Get the frequency of each n-gram
    n_gram_fd = nltk.FreqDist(ngrams(words, n_gram))

    return n_gram_fd.most_common()


def get_sentiment_score(raw_txt):
    """
    Calculates the sentiment score of the text

    :param raw_txt: the raw text of a file
    :return: a dictionary containing the sentiment score and standard error calculations
    """
    # Initialize VADER for sentiment analysis
    sentimentAnalyser = SentimentIntensityAnalyzer()

    # Get the compound sentiment score for the entire text
    sent_scores = sentimentAnalyser.polarity_scores(raw_txt)
    comp_score = sent_scores['compound']

    # Add score to dictionary; standard error is zero because sentimentAnalyzer analyzes an entire text rather than its
    # Individual Sentences
    sent_data = {'Text Sentiment': comp_score, 'Standard Error': 0}

    return sent_data


def json_parser(filename, n_gram, cnt_key_name="content"):
    """
    A custom parser for reading in and gathering data from a JSON file

    :param filename: the name of the JSON file
    :param cnt_key_name: the name of the key in the JSON file that will produce the entire text
    :return: a dictionary containing the data analysis results
    """
    # Load in the JSON file and extract the text
    with open(filename, encoding="utf-8") as json_file:
        raw = json.load(json_file)[0]
        text = raw[cnt_key_name]

        # Load in the list of stopwords
        stopwds = get_stopwords()

        # Create a list of the words in the JSON text
        lower_txt = str(text).lower()
        corp = re.sub('[^a-zA-Z]+', ' ', lower_txt).strip()
        tokens = word_tokenize(corp)
        words = [t for t in tokens if t not in stopwds]

        # Get the word counts for each word in the text
        wc = Counter(words)

        # Get the number of words in the text
        num = len(words)

        # Get the average word length for the text
        wd_lngths = [len(wd) for wd in words]
        avg_wl = round((sum(wd_lngths) / len(wd_lngths)), 3)

        # Get the n-gram freq counts
        ngram_data = get_ngram_freq(n_gram, words)

        # Get the sentiment data
        sentiment_data = get_sentiment_score(text)

        # Add these results to the results dictionary
        results = {'Wordcount':wc, 'NumWords':num, 'Avg. Word Length':avg_wl, 'Sentiment Data': sentiment_data,
                   '{n}-gram count'.format(n=str(n_gram)): ngram_data}

    return results
