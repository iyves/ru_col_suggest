import configparser
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


# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)

DOMAIN = config['SERVER']['DOMAIN']
HOST = config['SERVER']['HOST']
USER = config['SERVER']['USER']
PWD = config['SERVER']['PWD']


class CollocationAttestor:
    """Verifies that collocations exist in the cybercat database and calculates
    collocatiability scores
    """

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
        self.connection = mysql.connector.connect(host=HOST, database=DOMAIN, user=USER, password=PWD)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

    def execute_query(self, query: str):
        with self.connection.cursor() as cursor:
            return cursor.execute(query)

    def get_domain_size(self):
        query_result = self.execute_query(
            """
            SELECT COUNT(distinct lemma)
            FROM ruscorpora.lexicon;
            """
        ).fetchone()
        return query_result[0]

    def get_unigram_freqs(self, lemmas: Iterable):
        query_result = self.execute_query(
            """
            SELECT lemma_1, SUM(ngram_freq)
            FROM ruscorpora.ngrams_1_lex_mv
            WHERE lemma_1 in ('{}')
            GROUP BY lemma_1;
            """.format("','".join(lemmas))
        ).fetchall()
        return query_result

    def get_bigram_freqs(self, patterns: Iterable):
        query_result = self.execute_query(
            """
            SELECT CONCAT(lemma_1, " ", lemma_2), SUM(ngram_freq)
            FROM ruscorpora.ngrams_2_lex_mv
            WHERE CONCAT(lemma_1, " ", lemma_2) in ('{}')
            GROUP BY CONCAT(lemma_1, " ", lemma_2);
            """.format("','".join(patterns))
        ).fetchall()
        return query_result

    def attest_collocations(self, collocations: Iterable[str]) -> List[str]:
        pass

    def get_collocation_stats(self, collocations: Iterable[str]):
        pass
