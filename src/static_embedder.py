import configparser
import gensim
import logging
import math
import os
import sys

from enum import Enum
from gensim.models import FastText, KeyedVectors, Word2Vec
from itertools import product
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
    """The class for suggesting collocation via static word embeddings.
    """

    class Model(Enum):
        """ The type of static word embeddings to use.
        """
        WORD2VEC = 1
        FASTTEXT = 2
        GLOVE = 3

    def __init__(self, model_type, src: str):
        """Initialize the embedding model.

        :param model_type: The type of embeddings to use.
        :param src: The path to the location of the word embedding model.
        """
        self.model_type = model_type
        self.model = self.load_model(src)

    def load_model(self, src: str):
        if self.model_type == self.Model.WORD2VEC:
            return Word2Vec.load(src)
        elif self.model_type == self.Model.FASTTEXT:
            return FastText.load(src)
        elif self.model_type == self.Model.GLOVE:
            return KeyedVectors.load_word2vec_format(src, binary=True, unicode_errors='ignore')
        else:
            logging.error("Can only load w2v, fasttext, or glove models.")
            raise NotImplementedError

    def suggest_collocations(self, ngram_tokens: List[str], fixed_positions: List[int],
                             topn=10, cossim = True) -> List[Tuple[str, int, float]]:
        """Return a ranked list of collocations from a input sentence with fixed positions.

        :param ngram_tokens: A list of tokens with pos tags, forming the input sentence.
            ex. ["рассматривать_", "школа_N", "экономика_N"]
        :param fixed_positions: The tokens that will not be replaced.
            ex. [1, 0, 1] to only replace "школа"
        :param topn: The amount of tokens to suggest for each masked token.
        :param cossim: The similarity measure to use: True for cossim, False for cosmul. Default: cossim.
        :return a ranked list of suggested collocations.
            ex. [("рассматривать потребность экономика", 2, 0.123), ("рассматривать университет экономика", 6, 0.678)]
        """
        # Filter out the tokens for which replacements will be suggested
        if len(ngram_tokens) != len(fixed_positions):
            logging.error(f"Length of ngram_tokens ({len(ngram_tokens)}) is not equal to length of fixed_positions ({len(fixed_positions)})")
            return []
        tokens_to_replace = [ngram_tokens[i] for (i, fixed) in enumerate(fixed_positions) if fixed == 0]

        # W2v and GloVe models
        if self.model_type == self.Model.WORD2VEC or self.model_type == self.Model.GLOVE:
           # Ensure that the token is in the w2v vocabulary
           for token in tokens_to_replace:
               if token not in self.model.wv.vocab:
                   logging.error(f"'{token}' not in model vocabulary.")
                   return []
        elif self.model_type == self.Model.FASTTEXT:
            # No extra work to do for fastText model
            for token in tokens_to_replace:
                if token not in self.model.wv.key_to_index:
                    logging.info(f"'{token}' not in model vocabulary.")
        else:
            logging.error("Can suggest collocations for only w2v, fasttext, or glove models.")
            raise NotImplementedError
        # Get the similar words in format (token, pos tag, cosine distance)
        if cossim:
            similar_words = enumerate([tuple(token[0].split("_")) + (token[-1],)
                                       for token in self.model.wv.most_similar(positive=tokens_to_replace, topn=topn)])
        else:
            similar_words = enumerate([tuple(token[0].split("_")) + (token[-1],)
                                       for token in self.model.wv.most_similar_cosmul(positive=tokens_to_replace, topn=topn)])

        # Filter similar words by PoS tags in format (token, rank, cosine distance)
        suggested_replacements = []
        for ngram, fixed in zip(ngram_tokens, fixed_positions):
            if fixed == 1:
                suggested_replacements.append([tuple(ngram.split("_")[0], 0, 0)])
            else:
                # Get the similar words in format (token, pos tag, cosine distance)
                pos_tag = ngram.split("_")[1]
                suggested_replacements.append([
                    (token[1][0], token[0], token[1][2]) for token in similar_words
                        if token[1][1] == pos_tag
                ])

        # Flatten out the suggested_replacements list to format (colloc, rank)
        suggested_collocations = []
        for colloc_tuple in product(*suggested_replacements):
            colloc = []
            rank = 0
            cosine = 0
            for tuple in colloc_tuple:
                colloc.append(tuple[0])
                rank += tuple[1]
                cosine += tuple[2]
            suggested_collocations.append((" ".join(colloc), rank, cosine))
        return suggested_collocations
