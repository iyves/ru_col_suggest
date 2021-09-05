import configparser
import logging
import math
import mysql.connector
import os
import random
import sys

from enum import Enum
from itertools import chain
from pathlib import Path
from src.scripts.colloc import pmi, t_score
from typing import Iterable, List, Optional, Tuple


# For avoiding divide by zero errors
MINIMUM = .000000000001

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'collocatioin_attestor.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DOMAIN = config['SERVER']['DOMAIN']
HOST = config['SERVER']['HOST']
USER = config['SERVER']['USER']
PWD = config['SERVER']['PWD']


class CollocationAttestor:
    """Verifies that collocations exist in the cybercat database and calculates
    collocatiability scores
    """

    # Cache to avoid redundant db queries
    _domain_size = -1
    _collocation_stats = {}

    def __init__(self, domain=DOMAIN, host=HOST, user=USER, password=PWD):
        """Opens a connection with the cybercat database.
        :param domain: The domain name, default: use value from config.
        :param host: The IP address, default: use value from config.
        :param user: The username, default: use value from config.
        :param password: The password, default: use value from config.
        """
        self.domain = domain
        self.host = host
        self.user = user
        self.password = password

    def __enter__(self):
        """Creates a context manager for automatically opening and closing a connection
        to the cybercat db.
        """
        self.connection = mysql.connector.connect(host=self.host, database=self.domain, user=self.user, password=self.password)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

    def _execute_query(self, query: str):
        """Manages the context of a cursor to execute a query to the cybercat db.

        :param query: The SQL query as a string.
        :return The results of the query.
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            return result

    def get_domain_size(self):
        """Returns the amount of lemmas in the cybercat corpus."""
        if CollocationAttestor._domain_size == -1:
            query_result = self._execute_query(
                """
                SELECT COUNT(distinct lemma)
                FROM {}.lemmas;
                """.format(self.domain)
            )
            CollocationAttestor._domain_size = query_result[0][0]
        return CollocationAttestor._domain_size

    def get_frequency(self, ngrams: List[str]):
        """Returns the frequency of one or more lemmatized ngrams in the cybercat db.

        :param ngrams: The ngrams for which frequencies should be returned.
            Note: All ngrams must contain the same amount of tokens.
            Ex. ['что делать', 'рассматривать дело', 'новый метод']
        :return The frequencies of the inputted lemmas if they exist in the db.
        """
        if len(ngrams) == 0:
            return []
        n = len(ngrams[0].split(" "))
        uncached_ngrams= [ngram for ngram in ngrams
                          if ngram not in CollocationAttestor._collocation_stats]

        # Get the frequencies for uncached ngrams
        if len(uncached_ngrams) > 0:
            if n == 1:
                query_result = self._execute_query(
                    """
                    SELECT lemmas.lemma, SUM(uni.freq_all)
                    FROM
                        (SELECT id_lemmas, lemma 
                        FROM {1}.lemmas
                        WHERE hex(lemma) in (hex('{}'))
                        ) as lemmas
                    LEFT JOIN {1}.unigrams as uni ON uni.lemma = lemmas.id_lemmas
                    GROUP BY lemmas.lemma;
                    """.format("'), hex('".join(uncached_ngrams), self.domain)
                )
            elif n == 2:
                query_result = self._execute_query(
                    """
                    SELECT CONCAT(l1.lemma, " ", l2.lemma) as "bigram", SUM(bi_lemma.raw_frequency) as "frequency"
                    FROM 
                        (SELECT uni1.lemma as "lemma1", uni2.lemma as "lemma2", bi.raw_frequency
                        FROM {1}.2grams as bi
                        LEFT JOIN {1}.unigrams uni1 ON bi.wordform_1 = uni1.id_unigram
                        LEFT JOIN {1}.unigrams uni2 ON bi.wordform_2 = uni2.id_unigram
                        ) as bi_lemma
                    LEFT JOIN {1}.lemmas l1 ON bi_lemma.lemma1 = l1.id_lemmas
                    LEFT JOIN {1}.lemmas l2 ON bi_lemma.lemma2 = l2.id_lemmas
                    WHERE hex(CONCAT(l1.lemma, " ", l2.lemma)) in (hex('{0}'))
                    GROUP BY CONCAT(l1.lemma, " ", l2.lemma);
                    """.format("'), hex('".join(uncached_ngrams), self.domain)
                )
            elif n == 3:
                query_result = self._execute_query(
                    """
                    SELECT CONCAT(matches.lemma1, " ", matches.lemma2, " ", matches.lemma3) as trigram, SUM(3g_uni.raw_frequency) as freqency
                    FROM
                        (SELECT l1.lemma as "lemma1", l2.lemma as "lemma2", l3.lemma as "lemma3", tokens.id_trigram
                        FROM
                            (SELECT u1.lemma as t1, u2.lemma as t2, u3.lemma as t3, 3g.id_trigram as id_trigram 
                            FROM {1}.3grams_tokens as 3g
                            LEFT JOIN {1}.unigrams u1 ON 3g.w1 = u1.id_unigram
                            LEFT JOIN {1}.unigrams u2 ON 3g.w2 = u2.id_unigram
                            LEFT JOIN {1}.unigrams u3 ON 3g.w3 = u3.id_unigram
                            ) as tokens
                        LEFT JOIN {1}.lemmas l1 ON tokens.t1 = l1.id_lemmas
                        LEFT JOIN {1}.lemmas l2 ON tokens.t2 = l2.id_lemmas
                        LEFT JOIN {1}.lemmas l3 ON tokens.t3 = l3.id_lemmas
                        WHERE hex(CONCAT(l1.lemma, " ", l2.lemma, " ", l3.lemma)) in (hex('{0}'))
                        ) as matches
                    LEFT JOIN {1}.3grams 3g_uni ON matches.id_trigram = 3g_uni.id_trigram
                    GROUP BY CONCAT(matches.lemma1, " ", matches.lemma2, " ", matches.lemma3);
                    """.format("'), hex('".join(uncached_ngrams), self.domain)
                )
            else:
                logging.error("Can only get frequencies of 1/2/3-grams!")
                return []

            for row in query_result:
                lemma = row[0]
                frequency = row[1]
                CollocationAttestor._collocation_stats[lemma]= {}
                CollocationAttestor._collocation_stats[lemma]["freq"] = int(frequency)

            # Set frequency of unattested ngrams to 0
            for ngram in uncached_ngrams:
                if ngram not in CollocationAttestor._collocation_stats:
                    CollocationAttestor._collocation_stats[ngram] = {}
                    CollocationAttestor._collocation_stats[ngram]["freq"] = 0

        # Return the frequencies of the inputted ngrams
        frequencies = []
        for ngram in ngrams :
            frequencies.append([ngram, CollocationAttestor._collocation_stats[ngram]["freq"]])
        return frequencies

    def attest_collocations(self, lemmas: List[List[str]]):
        """Checks the cybercat database for the existence of either bigrams or
        trigrams.

        :param lemmas: Two lists (bigrams) or three lists (trigrams) of tokens
            to attest. Ex. [['мочь'], ['быть', 'делать']] for 'мочь быть' and 'мочь делать'
            or [['рассматривать'], ['потребность', 'школа'], ['экономика']] for
            'рассматривать потребность экономика' and 'рассматривать школа экономика'.
        :return The lemmatized bigrams or trigrams if they exist in the cybercat db.
        """
        # bigrams
        if len(lemmas) == 2:
            query = """SELECT DISTINCT l1.lemma as "lemma1", l2.lemma as "lemma2"
            FROM 
                (SELECT uni1.lemma as "lemma1", uni2.lemma as "lemma2"
                FROM {2}.2grams as bi
                LEFT JOIN {2}.unigrams uni1 ON bi.wordform_1 = uni1.id_unigram
                LEFT JOIN {2}.unigrams uni2 ON bi.wordform_2 = uni2.id_unigram
                ) as bi_lemma
            LEFT JOIN {2}.lemmas l1 ON bi_lemma.lemma1 = l1.id_lemmas
            LEFT JOIN {2}.lemmas l2 ON bi_lemma.lemma2 = l2.id_lemmas
            WHERE l1.lemma IN ('{0}') AND l2.lemma IN ('{1}');
            """.format("','".join(lemmas[0]), "','".join(lemmas[1]), self.domain)
        # trigrams
        elif len(lemmas) == 3:
            query = """
            SELECT DISTINCT l1.lemma as "lemma1", l2.lemma as "lemma2", l3.lemma as "lemma3"
            FROM
                (SELECT u1.lemma as t1, u2.lemma as t2, u3.lemma as t3
                FROM {3}.3grams_tokens as 3g
                LEFT JOIN {3}.unigrams u1 ON 3g.w1 = u1.id_unigram
                LEFT JOIN {3}.unigrams u2 ON 3g.w2 = u2.id_unigram
                LEFT JOIN {3}.unigrams u3 ON 3g.w3 = u3.id_unigram
                ) as tokens
            LEFT JOIN {3}.lemmas l1 ON tokens.t1 = l1.id_lemmas
            LEFT JOIN {3}.lemmas l2 ON tokens.t2 = l2.id_lemmas
            LEFT JOIN {3}.lemmas l3 ON tokens.t3 = l3.id_lemmas
            WHERE l1.lemma in ('{0}') AND l2.lemma IN ('{1}') AND l3.lemma IN ('{2}');
            """.format("','".join(lemmas[0]), "','".join(lemmas[1]), "','".join(lemmas[2]), self.domain)
        else:
            logging.error("Error: Can attest 2-grams or 3-grams only")
            return []

        query_result = self._execute_query(query)
        return [" ".join(result) for result in query_result]

    def get_collocation_stats(self, ngrams: List[str],
                              include_pmi=True, include_t_score=True,
                              include_ngram_freq=True):
        """Get the frequency and/or collocationability metrics for a list of
        collocations.

        :param ngrams: The ngrams for which stats should be returned.
            Note: All ngrams must contain the same amount of tokens.
            Ex. ['что делать', 'рассматривать дело', 'новый метод']
        :param include_pmi: Get the PMI score of all ngrams. Default: True.
        :param include_t_score: Get the t-score of all ngrams. Default: True.
        :param include_ngram_freq: Get the frequencies of all ngrams. Default: True.
        :return A dictionary where key is the collocation and value is a dictionary
            containing the included metrics {'pmi', 't_score', 'ngram_freq'}.
        """
        if len(ngrams) <= 0:
            return {}

        # Check if bigram or trigram
        gram = len(ngrams[0].split(" "))
        if gram == 2:
            bigrams = True
        elif gram == 3:
            bigrams = False
        else:
            logging.error("Can only get stats for 2/3-grams!")
            return {}

        # Initialize the ngram stats dictionary to return
        stats = {}

        # Load the frequencies for collocations
        for ngram, frequency in self.get_frequency(ngrams):
            stats[ngram] = {}
            if include_ngram_freq:
                stats[ngram]["ngram_freq"] = frequency

        # Calculate the pmi and t-scores
        if include_pmi or include_t_score:
            uncached_collocations = set()

            # Get the pmi and/or t_score for cached collocations
            for ngram in ngrams:
                if include_pmi:
                    if "pmi" not in CollocationAttestor._collocation_stats[ngram]:
                        uncached_collocations.add(ngram)
                    else:
                        stats[ngram]["pmi"] = CollocationAttestor._collocation_stats[ngram]["pmi"]
                if include_t_score:
                    if "t_score" not in CollocationAttestor._collocation_stats[ngram]:
                        uncached_collocations.add(ngram)
                    else:
                        stats[ngram]["t_score"] = CollocationAttestor._collocation_stats[ngram]["t_score"]

            # Get the pmi and/or t_score for uncached collocations
            if len(uncached_collocations) > 0:
                # Prepare the necessary information for calculating pmi and t_score
                patterns = []
                last_words = []
                pattern_dict = {}
                last_word_dict = {}

                for ngram in uncached_collocations:
                    split_collocation = ngram.split(" ")
                    # Get the pattern, aka the first n-1 tokens in an n-gram
                    if bigrams:
                        pattern = split_collocation[0]
                    else:
                        pattern = " ".join(split_collocation[:2])
                    patterns.append(pattern)
                    pattern_dict[pattern] = 0

                    last_word = split_collocation[-1]
                    last_words.append(last_word)
                    last_word_dict[last_word] = 0

                # Get the frequencies for patterns
                if bigrams:
                    for row in self.get_frequency(patterns):
                        pattern = row[0]
                        frequency = row[1]
                        pattern_dict[pattern] = frequency
                else:
                    for row in self.get_frequency(patterns):
                        pattern = row[0]
                        frequency = row[1]
                        pattern_dict[pattern] = frequency

                # Get the frequencies for last_words
                for row in self.get_frequency(last_words):
                    last_word = row[0]
                    frequency = row[1]
                    last_word_dict[last_word] = frequency

                for ngram in uncached_collocations:
                    split_collocation = ngram.split(" ")
                    if bigrams:
                        pattern = split_collocation[0]
                    else:
                        pattern = " ".join(split_collocation[:2])
                    last_word = split_collocation[-1]

                    colloc_count = CollocationAttestor._collocation_stats[ngram]["freq"]
                    c_pattern = pattern_dict[pattern]
                    c_lw = last_word_dict[last_word]

                    # Avoid divide by zero
                    if colloc_count == 0:
                        colloc_count = MINIMUM
                    if c_pattern == 0:
                        c_pattern = MINIMUM
                    if c_lw == 0:
                        c_lw = MINIMUM

                    if include_pmi:
                        pmi_value = pmi(colloc_count, c_pattern, c_lw, self.get_domain_size())
                        CollocationAttestor._collocation_stats[ngram]["pmi"] = pmi_value
                        stats[ngram]["pmi"] = pmi_value
                    if include_t_score:
                        t_score_value = t_score(colloc_count, c_pattern, c_lw, self.get_domain_size())
                        CollocationAttestor._collocation_stats[ngram]["t_score"] = t_score_value
                        stats[ngram]["t_score"] = t_score_value
        return stats
