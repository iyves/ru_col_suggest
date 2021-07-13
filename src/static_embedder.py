import configparser
import gensim
import logging
import math
import os
import sys

from enum import Enum
from gensim.models import FastText, Word2Vec
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'static_embedder.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DOMAIN = config['SERVER']['DOMAIN']
HOST = config['SERVER']['HOST']
USER = config['SERVER']['USER']
PWD = config['SERVER']['PWD']


class StaticEmbedder():
    class Model(Enum):
        WORD2VEC = 1
        FASTTEXT = 2
        GLOVE = 3

    def __init__(self, model_type, src: str):
        self.model_type = model_type
        self.model = self.load_model(src)
        pass

    def load_model(self, src: str):
        pass

    def suggest_collocations(self, ngram_tokens: List[str], fixed_positions: List[int]) \
            -> List[Tuple[str, int, float]]:
        pass
