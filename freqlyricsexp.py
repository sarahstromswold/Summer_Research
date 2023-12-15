# Most common kendrick and eminem lyrics

import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
assert sys.version_info >= (3, 7)
nltk.download('punkt')
nltk.download("stopwords")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist


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

# print(len(all_songs))

# string_data = str(data)
# print(all_songs[:100])
# print(string_data[:100])
# exit()
# example_string = "The box is really cool. The cat sat in it. It is shocking to realize how stupid Sarah can be. "\
# "Perhaps she will learn one day. I like to jog. Yesterday I went jogging, and the day before " \
# "that I jogged. Yesterday, I went jogging. shell shell shell shell shell shell shell"


if __name__ == "__main__":
    words_in_quote = word_tokenize(eminem_songs())
    stop_words = set(stopwords.words("english"))
    my_stop_words = {'\'', '\'s', 'n\'t', '\'m', '\'re', '\'ll','?', '.', ',', '’'}
    custom_words = my_stop_words.union(stop_words)
    filtered_list = []
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words_in_quote]
    for word in stemmed_words:
        if word.casefold() not in custom_words:
            filtered_list.append(word)
    # print(filtered_list)

    frequency_distribution = FreqDist(filtered_list).most_common(15)

    all_fdist = pd.Series(dict(frequency_distribution))

    # print(all_fdist)
    # Setting figure, ax into variables
    fig, ax = plt.subplots(figsize=(10, 10))

    # Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    all_plot = sns.barplot(x=all_fdist.index, y=all_fdist.values, ax=ax)
    plt.xticks(rotation=30)
    plt.show()
    # plt.savefig('output.png')




