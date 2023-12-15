# creates a frequency plot of the top 15 most common bigrams Kendrick and eminem uses

import nltk
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
assert sys.version_info >= (3, 7)
import json
nltk.download('punkt')
nltk.download("stopwords")
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist, bigrams
from nltk.stem import WordNetLemmatizer


def kendrick_songs():
    file = open('datasets.json')
    data = json.load(file)
    songs = data['train']

    all_songs = ""
    for s in songs:
        all_songs += s
    return all_songs


def eminem_songs():
    mat = np.genfromtxt("Eminem.csv", delimiter=",", dtype=str)
    s = mat[:, 6]
    total = ""
    for si in s:
        total += si
    return total


if __name__ == "__main__":

    words_in_quote = word_tokenize(kendrick_songs())
    stop_words = set(stopwords.words("english"))
    my_stop_words = {'\'', '\'s', 'n\'t', '\'m', '\'re', '\'ll','?', '.', ',', '’'}
    custom_words = my_stop_words.union(stop_words)
    filtered_list = []
    # stemmer = PorterStemmer()
    # stemmed_words = [stemmer.stem(word) for word in words_in_quote]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words_in_quote]

    for word in lemmatized_words:
        if word.casefold() not in custom_words:
            filtered_list.append(word)
    # print(filtered_list)
    test = bigrams(filtered_list)
    ngram_fd = FreqDist(test).most_common(15)
    # print(ngram_fd)

    ngram_joined = {'_'.join(k):v for k, v in ngram_fd}

    #my_dict = {}
    #for tup in ngram_fd:
    #    k = "_".join(tup[0])
    #    my_dict[k] = tup[1] the same thing as above

    # for k, v in ngram_joined.items():
        # print(k, "->", v) maps to which value

    # Convert to Pandas series for easy plotting
    ngram_freqdist = pd.Series(ngram_joined)

    # print(all_fdist)
    # Setting figure, ax into variables
    fig, ax = plt.subplots(figsize=(10, 10))

    # Setting plot to horizontal for easy viewing + setting title + display
    bar_plot = sns.barplot(x=ngram_freqdist.values, y=ngram_freqdist.index, orient='h', ax=ax)
    plt.title('Frequency Distribution')
    plt.show()
    # plt.savefig('output.png')



