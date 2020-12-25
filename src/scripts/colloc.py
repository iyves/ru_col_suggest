"""From https://github.com/SerasLain/CAT-collocations/blob/master/colloc.py
"""

from math import log10, sqrt, log2


def read_ngrams(path):
    """
    Loads ngrams from csv
    :param path: str, path to the tab-separated csv-file with ngrams
    :return: container: dict, dictionary of mwe with its frequencies.
    """

    file = open(path, 'r', encoding='utf-8')
    container = {}
    for line in file:
        collocation_tags, freq = line.strip('\n').split('\t')
        container[collocation_tags] = freq
    return container


def measure(colloc_tag, colloc_count, n1grams, unigrams, n):
    """
    counts scores for all measures
    :param colloc_tag:str key from dictionary of ngrams like 'word word/tag tag'
    :param colloc_count: int, how many times collocation appears
    :param n1grams: dict, a dict of MWE n-1-grams, like bigrams
    :param unigrams: dict, a dict of words with its count in the corpus
    :param n: int, len of corpus
    """
    collocation_words, collocation_tags = colloc_tag.split('/')
    collocation_words = collocation_words.split(' ')
    collocation_tags = collocation_tags.split(' ')
    pattern_words = ' '.join(collocation_words[:-1])
    pattern_tags = ' '.join(collocation_tags[:-1])

    pattern = pattern_words + '/' + pattern_tags
    last_word = collocation_words[-1] + '/' + collocation_tags[-1]
    c_pattern = int(n1grams[pattern])
    c_lw = int(unigrams[last_word])
    colloc_count = int(colloc_count)
    if c_pattern == 0:
        c_pattern = 1

    tsc = t_score(colloc_count, c_pattern, c_lw, n)
    pmisc = pmi(colloc_count, c_pattern, c_lw, n)
    logdsc = logDice(colloc_count, c_pattern, c_lw)
    return logdsc, pmisc, tsc


def t_score(colloc_count, c_pattern, c_lw, n):
    """
    t-score measure for collocation based on the t-test. [Church, Using statistics in lexical analysis]
    t-score = (colloc_count - (c_pattern * c_lw) / n) / (sqrt(colloc_count))
    :param colloc_count:int how many times colloc appears in corpus
    :param c_pattern:int how many times all words but last appear in corpus
    :param c_lw:int how many times last word appears in corpus
    :param n:int corpus size
    :return: t-score for the collocation
    """
    score = float((colloc_count - (c_pattern * c_lw) / n)) / (sqrt(colloc_count))
    return score


def pmi(colloc_count, c_pattern, c_lw, n):
    """
    PMI-measure for collocation. PMI = log10((N*c(w1, w2, w3)/(c(pattern)*c(last_word)))
    [Foundations of Statistical Natural Language Processing]
    c(w) is how many times this word (or these words) appears in corpus
    pattern is a collocation without a last word
    N is a number of tokens in corpus
    :param colloc_count:int how many times collocation appears in corpus
    :param c_pattern:int how many times all words but last appear in corpus
    :param c_lw:int how many times last word appears in corpus
    :param n:int the number of tokens in corpus
    :return: float, pmi score for the collocation
    """
    score = log10((n * colloc_count / (c_pattern * c_lw)))
    return score


def logDice(colloc_count, c_pattern, c_lw):
    """
    LogDice measure.
    [Rychly, A Lexicographer-Friendly Association Score]
    :param colloc_count:int how many times collocation appears in corpus
    :param c_pattern:int how many times all words but last appear in corpus
    :param c_lw:int how many times last word appears in corpus
    :return: float
    """
    score = 14 + log2(2*colloc_count / (c_lw + c_pattern))
    return score


def c_value(colloc_l, colloc_freq, longer_c, longer_freq):
    """
    C-value measure.
    :param colloc_freq: frequency of the collocation in corpus
    :param colloc_l: length of the collocation
    :param longer_freq: how many times longer collocations with this ngram appear in corpus
    :param longer_c: how many different collocations with this ngram are in corpus
    :return: c_value, float
    """
    longer_c = int(longer_c)
    c_val = (colloc_l - 1) * (colloc_freq - (longer_freq / longer_c))
    return c_val