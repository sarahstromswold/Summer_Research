# freq plot for the top 15 most common POS for kendrick and eminem

import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
assert sys.version_info >= (3, 7)
import json
nltk.download('punkt')
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist, RegexpTokenizer
from collections import Counter
from string import punctuation


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
    lower_case = eminem_songs().lower()
    tokens = nltk.word_tokenize(lower_case)
    tags = nltk.pos_tag(tokens)

    counts = Counter(tag for word,  tag in tags)
    print(counts)
    # print(tags)
    counts = {k: counts[k] for k, v in counts.items() if not any(p in k for p in punctuation)}
    print(counts)
    frequency_distribution = FreqDist(counts).most_common(15)

    all_fdist = pd.Series(dict(frequency_distribution))

    # print(all_fdist)
    # Setting figure, ax into variables
    fig, ax = plt.subplots(figsize=(10, 10))

    # Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    all_plot = sns.barplot(x=all_fdist.index, y=all_fdist.values, ax=ax)
    plt.xticks(rotation=30)
    plt.show()
    # plt.savefig('output.png')
