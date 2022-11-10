"""
Mia Huebscher

A reusable library that utilizes NLP to create visualizations using text data
"""
import re
import requests
import nltk
import random as rnd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from collections import defaultdict, Counter
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sankey_funcs import code_mapping, make_sankey


class AnalyzeText:

    def __init__(self):
        """Constructor"""
        # Extract data (state)
        self.data = defaultdict(dict)

    def _save_results(self, label, results):
        """
        Saves the data results of each text into the data dictionary held in self
        :param label: the name of the file that results are extracted from
        :param results: a dictionary containing the text's data
        :return:
        """
        for key, val in results.items():
            self.data[key][label] = val

    @staticmethod
    def rmv_stop_words(full_txt, stpwds_lnk=("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b" +
                                             "5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt")):
        """
        Removes the stopwords from a text and creates a list containing the words of this filtered text

        :param full_txt: all the text from a given file
        :param stpwds_lnk: a link to the site where additional stopwords are held
        :return: a list containing the words of the filtered text
        """
        # Load in the stopwords from the NLTK library
        nltk_stp_wds = stopwords.words('english')

        # Load in more stopwords from the web
        stopwords_list = requests.get(stpwds_lnk).content
        stopwds = list(set(stopwords_list.decode().splitlines()))

        # Add the list of stopwords from the NLTK library
        stopwds += nltk_stp_wds

        # Create a list of the words in the text that are not stopwords
        txt = str(full_txt).lower()
        corp = re.sub('[^a-zA-Z]+', ' ', txt).strip()
        tokens = word_tokenize(corp)
        words = [t for t in tokens if t not in stopwds]

        return words

    @staticmethod
    def get_sentiment_score(full_txt):
        """
        Calculates the sentiment score of the text and also the standard error of this calculation

        :param lst_of_txt: a list of all the sentences in a text
        :return: a dictionary containing the sentiment score and standard error calculations
        """

        # Initialize VADER for sentiment analysis
        sentimentAnalyser = SentimentIntensityAnalyzer()

        # Get the compound sentiment of the full text
        sent_scores = sentimentAnalyser.polarity_scores(full_txt)
        comp_score = sent_scores['compound']

        # Create a dictionary with this data
        sent_data = {'Text Sentiment': comp_score}

        return sent_data

    @staticmethod
    def get_ngram_data(n_gram, words):
        """
        Finds the n_gram frequencies of the text and returns data as a list of tuples

        :param n_gram: value that decides number of contiguous items to get frequency of
        :param words: the list of words from the text, excluding stopwords
        :return: a list of tuples (n-gram, frequency) sorted by frequency
        """

        # Get the frequency of each n-gram
        n_gram_fd = nltk.FreqDist(ngrams(words, n_gram))

        return n_gram_fd.most_common()

    @staticmethod
    def get_results(words, sent_data, n_gram_data, n_gram):
        """
        Consolidates all the results of data analysis into a single dictionary

        :param words: a list of words in the text that are not stopwords
        :param sent_data: a dictionary of sentiment data
        :param n_gram_data: a list of tuples containing (n-gram, frequency in text)
        :param n_gram: the number the user wants n-gram to be
        :return: a dictionary containing all the results of a text's analysis
        """
        # Create an empty dictionary
        results = {}

        # Get the word counts and add them to the results dictionary
        wc = Counter(words)
        results['Wordcount'] = wc

        # Get the number of words in the text and add it to the results dictionary
        results['NumWords'] = len(words)

        # Get the average length of words in the text and add value to results dictionary
        txt_wd_lngths = [len(wd) for wd in words]
        results['Avg. Word Length'] = round((sum(txt_wd_lngths) / len(txt_wd_lngths)), 3)

        # Add the sentiment score data to the results dictionary
        results['Sentiment Data'] = sent_data

        # Add the n_gram data to the results dictionary
        results['{n}-gram count'.format(n=str(n_gram))] = n_gram_data

        return results

    @staticmethod
    def _default_parser(filename, n_gram):
        """
        Processes a generic text file

        :param filename: the name of the file to open
        :param n_gram: the number that the user wants n_grams to include
        :return: the results of the text file's analysis
        """

        # Open file and read it in
        with open(filename, encoding='utf8') as txt_file:
            lst_of_txt = txt_file.readlines()

            # Create an empty string to include the parsed text of the file
            full_txt = ''

            # Add each sentence in lst_of_txt to the full text
            for sent in lst_of_txt:
                full_txt += sent

            # Remove the stopwords from the text
            words = AnalyzeText.rmv_stop_words(full_txt)

            # Get the sentiment score data for the text
            sent_data = AnalyzeText.get_sentiment_score(full_txt)

            # Get the n-gram frequency data for the text
            n_gram_data = AnalyzeText.get_ngram_data(n_gram, words)

            # Create a dictionary containing the results of analysis
            results = AnalyzeText.get_results(words, sent_data, n_gram_data, n_gram)

        return results

    def load_text(self, filename, label=None, parser=None, cnt_key_name=None, n_gram=2):
        """
        Registers the text file with the NLP framework
        
        :param filename: the name of the file to open
        :param label: the name the user wants the file to be identified with
        :param parser: optional parameter in case the given file cannot be processed with the default txt parser
        :param cnt_key_name: the key value that corresponds to the full text in a json file (optional)
        :param n_gram: the number of words the user wants the n_gram to contain
        :return: 
        """

        if parser is None:
            results = AnalyzeText._default_parser(filename, n_gram)

        else:
            if cnt_key_name is None:
                results = parser(filename, n_gram)
            else:
                results = parser(filename, n_gram, cnt_key_name)

        if label is None:
            label = filename

        self._save_results(label, results)

    @staticmethod
    def get_sankey_data(txt_data, word_list, k):
        """
        Obtains the necessary data to plot the sankey diagram

        :param txt_data: the data held in self
        :param word_list: a list of words the user wants the sankey to show
        :param k: the number of words the user wants the sankey to match to each file of text
        :return: a dataframe with the data for the sankey
        """
        # Create an empty Dataframe
        txts_df = pd.DataFrame()

        # A list of all the labels for the files of texts given
        sources = list(txt_data['Avg. Word Length'].keys())

        # If a word list is given, add relevant data to the dataframe based on the words in that list
        if word_list:
            # Create lists for the targets and values for the sankey
            source_lst = []
            targets = []
            values = []

            # Update these lists to include relevant data
            for wd in word_list:
                for source in sources:
                    for word, count in txt_data['Wordcount'][source].items():
                        if wd.lower() == word.lower():
                            source_lst.append(source)
                            targets.append(wd)
                            values.append(count)
                        else:
                            pass

            txts_df['sources'] = source_lst

        # If a word list is not given, have the sankey include the k most common words from each file of text
        else:
            # Add a sources column to the dataframe
            txts_df['sources'] = sources * k

            # Obtain the k most common words from each file of text
            kcommon_wds = []
            for source in sources:
                kcommon_wds.append(txt_data['Wordcount'][source].most_common(k))

            # Initialize the targets and values lists
            targets = []
            values = []

            # Update the targets and values lists to include relevant values
            count = 0
            while count < k:
                for big_lst in kcommon_wds:
                    targets.append(big_lst[count][0])
                    values.append(big_lst[count][1])
                count += 1

        # Add a targets and a values column to the Dataframe
        txts_df['targets'] = targets
        txts_df['values'] = values

        return txts_df

    def wordcount_sankey(self, word_list=None, k=5):
        """
        Plots a sankey showing specific words paired to their text sources, with link width determined by word count in
        each text

        :param word_list: a list of words the user wants the sankey to include
        :param k: the number of words the user wants each text file to be connected to
        :return:
        """
        # Get the Dataframe of data for the sankey
        sankey_df = AnalyzeText.get_sankey_data(self.data, word_list, k)

        if word_list:
            inserts = ''
            if len(word_list) == 1:
                inserts += str(word_list[0].capitalize()) + ' Appears'
            else:
                for word in word_list:
                    if word_list.index(word) != (len(word_list) - 1):
                        inserts += str(word.capitalize()) + ', '
                    else:
                        inserts += 'and ' + str(word.capitalize()) + ' Appear'

            title = f'The Number of Times {inserts} in Each Text'
        else:
            # Create a generic title for the sankey
            sankey_title='The {kvalue} Most Common Words in {filenum} Different Texts and Their Frequencies Among the '\
                         'Texts'

            # Manipulate the generic title based on the user's inputs
            title = sankey_title.format(kvalue=str(k), filenum=str(len(list(self.data['Avg. Word Length'].keys()))))

        # Make the sankey
        make_sankey(sankey_df, 'sources', 'targets', title, 'values', pad=2)

    def plot_wd_lens(self, wd_len_dct, colors):
        """
        Plots the average length of words in each text

        :param wd_len_dct: a dictioanry matching texts to their average length of words
        :param colors: the colors the user wants the bars to be plotted with
        :return:
        """
        # Get the x and y values
        yvals = list(wd_len_dct.keys())
        xvals = list(wd_len_dct.values())

        # Plot the bars using the x and y values based on whether color is None
        if colors is None:
            bars = plt.barh(np.arange(len(yvals)), xvals)
        else:
            bars = plt.barh(np.arange(len(yvals)), xvals, color=colors)

        # Add the average word lengths for each text on top of each the bars
        plt.bar_label(bars)

        # Customize the bar plot
        max_x = max(xvals)
        plt.xticks(np.arange(0, max_x+1, step=0.5))
        plt.yticks(np.arange(len(yvals)), yvals)
        plt.title('Average Length of the Words in {n} Separate Texts'.format(n=str(len(list(wd_len_dct)))))
        plt.ylabel('Sources of Texts')
        plt.xlabel('Average Word Length in Text')

        return bars

    def plot_sentiments(self, sentiment_data, colors):
        """
        Plots a bar chart comparing the sentiment of each text along with standard error bars

        :param sentiment_data: a dictionary with sentiment average and standard error
        :param colors: the colors the user wants the bars to be plotted with
        :return:
        """
        # Collect the data
        x_vals = list(sentiment_data.keys())
        y_vals = []

        for dct in list(sentiment_data.values()):
            y_vals.append(dct['Text Sentiment'])

        # Plot the data and add features to the plot
        bars = plt.bar(x_vals, y_vals, color=colors)
        plt.bar_label(bars)
        plt.title('The Sentiment Scores of {n} Separate Texts'.format(n=len(x_vals)))
        plt.ylabel('Sentiment Score [-1,1]')
        plt.xlabel('Sources of Texts')

    def get_maxmin_sent(self):
        """
        Get the two texts with the largest difference in sentiment

        :return: a dictionary of the texts with the largest difference in sentiment
        """
        # Initialize a dictionary
        maxmin_dct = {}

        # Get the sentiment data from self
        sent_dct = self.data['Sentiment Data']

        # Get a list of sentiment averages
        sent_avgs = [sent_dct[source]['Text Sentiment'] for source in sent_dct.keys()]

        # Get the labels for the texts with the max and min sentiments
        max_label = list(sent_dct.keys())[sent_avgs.index(max(sent_avgs))]
        min_label = list(sent_dct.keys())[sent_avgs.index(min(sent_avgs))]

        # Add these values to the dictionary
        maxmin_dct['Max Sentiment'] = max_label
        maxmin_dct['Min Sentiment'] = min_label

        return maxmin_dct

    def plot_venn_diagram(self):
        """
        Gets the two articles with the most opposing sentiment scores (biggest distance between scores) and creates
        a venn diagram comparing the words in each text
        :return:
        """
        # Get the word count frequencies
        word_counts = self.data['Wordcount']

        # Find texts with the largest distance between sentiment scores
        maxmin_dct = self.get_maxmin_sent()

        # Extract the labels of the texts
        max_label = maxmin_dct['Max Sentiment']
        min_label = maxmin_dct['Min Sentiment']

        # Create subsets for both texts that contain the words in the text
        xy_val_dct = {}
        for label in [max_label, min_label]:
            tup_lst = word_counts[label].most_common()
            x_vals = []
            for tuple in tup_lst:
                x_vals.append(tuple[0])
            xy_val_dct[label] = x_vals

        # Plot the venn diagram
        venn2(subsets=[set(xy_val_dct[max_label]), set(xy_val_dct[min_label])], set_labels=[max_label, min_label],
              set_colors=('green', 'red'))

    def plot_ngrams_wc(self, wordcloud_file_lable):
        """
        Creates a wordcloud using the user-given text's n_gram counts

        :param wordcloud_file_lable: the name of the file the user wants a wordcloud for
        :return:
        """

        # Extract the data from self that has the n_gram counts
        for key in self.data.keys():
            if 'gram' in key:
                full_ngram_data = self.data[key]

        # Extract the n_gram counts for the user-given file
        file_ngram_data = full_ngram_data[wordcloud_file_lable]

        # Create a string that includes each n_gram in the n_gram data separated by a space
        ngram_txt = ''
        for tuple in file_ngram_data:
            ngrams = tuple[0]
            ngram = ''
            for word in ngrams:
                ngram += word + '_'

            ngram = ngram[:-1]
            ngram_txt += ngram + ' '

        # Plot the wordcloud and customize it
        wordcloud = WordCloud(width=2000, height=1334, random_state=1, background_color='black', colormap='Pastel1',
                              max_words=75, collocations=False, normalize_plurals=False).generate(ngram_txt)
        plt.imshow(wordcloud)
        plt.title('A Word Cloud Showing the Frequency of {n}-grams in {source}'.format(n=list(self.data.keys())[-1][0],
                                                                                      source=wordcloud_file_lable))
        plt.axis('off')

    def txt_feature_comparisons(self, wordcloud_file_lable, colors=None):
        """
        Creates a series of subplots that compares the sentiment, average word length, and average number of words
        between the given data files. Also includes a subplot depicting a word cloud.

        :param wordcloud_file_lable: the file label the user wants a wordcloud for
        :param colors: the colors the user wants each text's data to be colored in
        :return: a series of subplot comparing the features of the given data files
        """
        # Create a figure that will contain four plots
        plt.subplots(2, 2)

        # Get the data for subplot 1, a horizontal bar chart comparison for word length
        wd_len_dct = self.data['Avg. Word Length']

        # Plot the data
        plt.subplot(2,2,1)
        self.plot_wd_lens(wd_len_dct, colors)

        # Get the data for subplot 2, a vertical bar chart comparison for sentiment scores and standard errors
        sentiment_data = self.data['Sentiment Data']

        # Plot the data
        plt.subplot(2,2,2)
        self.plot_sentiments(sentiment_data, colors)

        # Plot a venn diagram comparing the words used in two texts that have the largest difference in sentiment
        plt.subplot(2,2,3)
        plt.title('A Venn Diagram Comparing Word Usage in the Two Texts with the Largest Difference in Sentiment Score',
                  fontsize=10, wrap=True, va='top')
        self.plot_venn_diagram()

        # Plot a wordcloud created with n_gram data from a user-specified file
        plt.subplot(2,2,4)
        self.plot_ngrams_wc(wordcloud_file_lable)

        # Show the graphs
        plt.subplots_adjust(left=0.13,bottom=0.07,right=0.96,top=0.93,wspace=0.4,hspace=0.4)
        plt.show()

    def plot_common_words(self, colors=None, k=10):
        """
        Creates a stacked bar chart comparing the k most common words in each text file

        :param colors: the colors the user wants the bars to be plotted with
        :param k: the number of most common words the user wants to compare across texts
        :return:
        """
        # Get the data for word counts and extract the source text labels
        word_counts = self.data['Wordcount']
        labels = list(word_counts.keys())

        # Create a dictionary that pairs each text to a tuple containing (the k most common words in the text,
        # Their word counts in the text)
        xy_val_dct = {}
        for label in labels:
            tup_lst = word_counts[label].most_common(k)
            y_vals = []
            x_vals = []
            for tuple in tup_lst:
                x_vals.append(tuple[0])
                y_vals.append(tuple[1])
            xy_val_dct[label] = (x_vals, y_vals)

        # Create a list of all the words in every text, excluding repeats
        all_wds = []
        for key, value in xy_val_dct.items():
            for wd in value[0]:
                if wd not in all_wds:
                    all_wds.append(wd)

        # For each text, get the y values for its bar and add it to a bigger list
        big_lst_of_ys = []
        for label in labels:
            y = []
            for word in all_wds:
                if word in xy_val_dct[label][0]:
                    y_vals_lst = xy_val_dct[label][1]
                    y.append(y_vals_lst[xy_val_dct[label][0].index(word)])
                else:
                    y.append(0)
            big_lst_of_ys.append(y)

        # For every list of y values in the big list, plot the y values with the corresponding text label and color in
        # A stacked bar chart
        for idx in range(len(big_lst_of_ys)):
            if idx != 0:
                bottoms_lst = big_lst_of_ys[:idx]
                bottoms = [sum(x) for x in zip(*bottoms_lst)]
                if colors:
                    plt.bar(all_wds, big_lst_of_ys[idx], bottom=np.array(bottoms), label=labels[idx], color=colors[idx])
                else:
                    plt.bar(all_wds, big_lst_of_ys[idx], bottom=np.array(bottoms), label=labels[idx])
            else:
                if colors:
                    plt.bar(all_wds, big_lst_of_ys[idx], label=labels[idx], color=colors[idx])
                else:
                    plt.bar(all_wds, big_lst_of_ys[idx], label=labels[idx])

        # Customize the stacked bar plot
        plt.title('The {n} Most Common Words in Each Text and Their Frequency Count in Each Text'.format(n=k),
                  fontsize=15, wrap=True)
        plt.xticks(rotation=90)
        plt.xlabel('Common Words')
        ot = big_lst_of_ys.pop()
        tops = [sum(x) for x in zip(*[bottoms, ot])]
        plt.ylim(0, max(tops) + 5)
        plt.ylabel('Frequency Count in Text')
        plt.legend(title='Text Source')
