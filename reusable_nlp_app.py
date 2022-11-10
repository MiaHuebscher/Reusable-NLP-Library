from reusable_nlp import AnalyzeText
import nlp_parsers as nlpp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Create the class object
    at = AnalyzeText()

    # Load the texts that will be analyzed
    at.load_text('wsj.txt', 'The Wall Street Journal', n_gram=3)
    at.load_text('msnbc.txt', 'MSNBC', n_gram=3)
    at.load_text('Federalist.txt', 'The Federalist', n_gram=3)
    at.load_text('foxnews.json', 'Fox News', parser=nlpp.json_parser, n_gram=3)

    # Plot a sankey diagram showing the 10 most common words in each text and their frequencies among the texts
    at.wordcount_sankey(k=10)

    # Plot a series of four subplots that compare the attributes of each text; also include a subplot for a wordcloud
    # Of bi-grams
    at.txt_feature_comparisons(wordcloud_file_lable='Fox News', colors=['cornflowerblue','royalblue','lightcoral','red'])

    # Plot a stacked bar chart comparing the 10 most common words in each text
    plt.figure(2)
    at.plot_common_words(colors=['cornflowerblue', 'royalblue', 'lightcoral', 'red'], k=15)
    plt.tight_layout()
    plt.show()

