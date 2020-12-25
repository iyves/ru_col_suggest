import configparser
import gensim
import logging
import mysql.connector
import pickle
import os
import random
import sys
import wget
import zipfile

from enum import Enum
from gensim.models import FastText, Word2Vec
from itertools import chain
from pathlib import Path
from src.helpers import get_text, get_file_names
from typing import Iterable, List, Optional, Tuple


# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'train_word_embeddings.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DOMAIN = config['SERVER']['DOMAIN']
HOST = config['SERVER']['HOST']
USER = config['SERVER']['USER']
PWD = config['SERVER']['PWD']


def pickle_loader(pickle_file):
    try:
        while True:
            yield pickle.load(pickle_file)
    except EOFError:
        pass


class SentencesIterator():
    def __init__(self, generator_function, source):
        """Helper class for the WordEmbedder::load_corpus function.

        :param generator_function: The generator function (load_corpus).
        :param source: The folder containing the preprocessed corpus pickle
            files.
        """
        self.source = source
        self.generator_function = generator_function
        self.generator = self.generator_function(self.source)

    def __iter__(self):
        # reset the generator
        self.generator = self.generator_function(self.source)
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result


class WordEmbedder:
    """Deals with downloading, training, and utilizing word embedding models.
    """

    class Model(Enum):
        """Pre-trained lemma-based word embedding models from Kutuzov's site.
        https://rusvectores.org/ru/models/
        """
        word2vec = 'http://vectors.nlpl.eu/repository/20/180.zip'
        fastText = 'http://vectors.nlpl.eu/repository/20/213.zip'

    punctuation = r"!\"#$%&'()*+,-.\/:;<=>?@[\]^_`{|}~ʹ…〈〉«»—„“"

    @staticmethod
    def filter_results(lemmas: List[List[str]]):
        """

        :param lemmas:
        :return:
        """
        if len(lemmas) == 2:
            query = "SELECT distinct ng2.lemma_1, ng2.lemma_2, SUM(ng2.ngram_freq), SUM(ng2.doc_freq) " \
                    "FROM ngrams_2_lex_mv as ng2 " \
                    "WHERE ng2.lemma_1 in ('{}') and ng2.lemma_2 in ('{}') " \
                    "GROUP BY ng2.lemma_1, ng2.lemma_2" \
                .format("','".join(lemmas[0]), "','".join(lemmas[1]))
        elif len(lemmas) == 3:
            query = "SELECT distinct ng3.lemma_1, ng3.lemma_2, ng3.lemma_3, SUM(ng3.ngram_freq), SUM(ng3.doc_freq) " \
                    "FROM ngrams_3_lex_mv as ng3 " \
                    "WHERE ng3.lemma_1 in ('{}') and ng3.lemma_2 in ('{}') and ng3.lemma_3 in ('{}')" \
                    "GROUP BY ng3.lemma_1, ng3.lemma_2, ng3.lemma_3" \
                .format("','".join(lemmas[0]), "','".join(lemmas[1]), "','".join(lemmas[2]))
        else:
            logging.error("Error: Can filter results for 2-grams or 3-grams only")
            return []

        connection = mysql.connector.connect(host=HOST, database=DOMAIN, user=USER, password=PWD)
        cursor = connection.cursor()
        cursor.execute(query)
        query_result = cursor.fetchall()
        cursor.close()
        connection.close()
        return query_result

    def __init__(self, model_type: Model):
        self.model_type = model_type
        self.model = None

    def download_model(self, target_dir: str, filename: str = None) -> None:
        """Downloads Kutuzov's trained word embedding model.

        :param target_dir: The directory to which the model will be downloaded.
        :param filename: The filename under which to save the model.
        """
        if filename is None:
            filename = self.model_type.value.split('/')[-1]
        model_path = str(Path(target_dir, filename))
        if not os.path.isfile(model_path):
            print('Model at "{}"" not found. Downloading...'.format(model_path), file=sys.stderr)
            wget.download(self.model_type.value, out=model_path)

        print('\nLoading the model from {}...'.format(model_path), file=sys.stderr)
        with zipfile.ZipFile(model_path, 'r') as archive:
            stream = archive.open('model.bin')
            self.model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)

    def _load_sentences(self, corpus):
        """Load sentences from a preprocessed corpus.

        :param corpus: The corpus in LISTS_OF_PAGES_AND_PARAGRAPHS_AND_SENTENCES
            format.
        :return: A generator that iterates through the sentences in a corpus.
        """
        for page in corpus:
            for paragraph in page:
                for sentence in paragraph:
                    yield [s for s in sentence if not "PUNCT" in s
                           and all([char not in s.split("_")[0] for char in self.punctuation])]

    def load_corpus(self, source: str):
        """Prepare the preprocessed corpus and return each sentence.

        :param source: The folder containing the preprocessed corpus pickle
            files.
        :return: A generator that iterates through all preprocessed sentences.
        """
        files = get_file_names(source, ".pickle")
        for file in files:
            with open(file, "rb") as f:
                for sentence in self._load_sentences(pickle.load(f)):
                    yield sentence

    def train_subcorpora(self, pickles_dir: str, source_folders: Iterable[str],
                         target_dir: str, target_filenames: Iterable[str] = None) -> bool:
        """Train one or more model on one or more subcorpus.

        :param pickles_dir: The root directory that contains the directories
            with the preprocessed text pickle files.
        :param source_folders: The directories with the preprocessed text
            pickle files.
        :param target_dir: The root directory to save the trained models.
        :param target_filenames: The filename by which to save the models.
        :return: True if the training completed successfully, otherwise False.
        """
        if target_filenames is None:
            target_filenames = []

        for i, foldername in enumerate(source_folders):
            filename = "{}.model".format(foldername)
            if i >= len(target_filenames):
                filename = target_filenames[i]

            model_path = str(Path(target_dir, filename))
            source = str(Path(pickles_dir, foldername).resolve())
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # Set up the article to preprocess
            files = get_file_names(source, ".pickle")
            random.shuffle(files)
            sentences = SentencesIterator(self.load_corpus, source)

            if self.model_type == self.Model.word2vec:
                model_w2v = Word2Vec(sentences, size=100, window=10, min_count=5)
                model_w2v.save(model_path)
            elif self.model_type == self.Model.fastText:
                model_fasttest = FastText(sentences, size=100, window=10, min_count=5)
                model_fasttest.save(model_path)
            else:
                logging.error("ERROR: {} model not supported".format(model_path))
                return False
        return True

    def train_corpus(self, pickles_dir: str, source_folders: Iterable[str],
                     target_dir: str, target_filename: str = "all.model", params = None) -> bool:
        """Train the model on the entire corpus.

        :param pickles_dir: The root directory that contains the directories
            with the preprocessed text pickle files.
        :param source_folders: The directories with the preprocessed text
            pickle files.
        :param target_dir: The root directory to save the trained model.
        :param target_filename: The filename by which to save the model.
        :param params: The options to pass onto gensim for training.
        :return: True if the training completed successfully, otherwise False.
        """
        # Set defaults for training if not specified
        if params is None:
            params = {
                "size": 100,
                "window": 10,
                "min_count": 10
            }

        # Set up the article to preprocess
        files = []
        for folder in source_folders:
            source = str(Path(pickles_dir, folder).resolve())
            files += get_file_names(source, ".pickle")
        random.shuffle(files)

        model_path = str(Path(target_dir, target_filename))
        sentences = SentencesIterator(self.load_corpus, source)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if self.model_type == self.Model.word2vec:
            self.model = Word2Vec(sentences, **params)
            self.model.save(model_path)
        elif self.model_type == self.Model.fastText:
            self.model = FastText(sentences, **params)
            self.model.save(model_path)
        else:
            logging.error("ERROR: {} model not supported".format(model_path))
            return False
        return True

    def load_model(self, model_path: str) -> bool:
        """Load a pre-trained model.

        :param model_path: The path to the model to be loaded.
        :return: True is the model was successfully loaded, otherwise False.
        """
        if self.model_type == self.Model.word2vec:
            self.model = gensim.models.Word2Vec.load(model_path)
        elif self.model_type == self.Model.fastText:
            self.model = gensim.models.FastText.load(model_path)
        else:
            logging.error("ERROR: {} model not supported".format(model_path))
            return False
        return True

    def check_loaded(self) -> bool:
        """Ensures that a model is loaded. Logs an error if not.

        :return: True if a model is loaded, otherwise False.
        """
        if self.model is None:
            logging.error("Error: Failed to predict similar, because no model is loaded.")
            return False
        return True

    def print_vocabulary(self, limit: int = -1) -> bool:
        """Prints the vocabulary of the model.

        :param limit: The maximum amount of vocabulary terms to print. -1 to
            print the entire vocabulary.
        :return: True if the vocabulary was printed, otherwise False.
        """
        if not self.check_loaded():
            return False

        for i, word in enumerate(self.model.wv.vocab):
            if i == limit:
                break
            print(word)
        return True

    def word_in_vocabulary(self, word: str) -> bool:
        """Checks if a term is in the vocabulary of this model.

        :param word: The term to check.
        :return: True if the term is in the vocabulary, otherwise False.
        """
        if not self.check_loaded():
            return False
        return word in self.model.wv.key_to_index

    # Universal POS tags:
    ## ADJ ADP PUNCT ADV AUX SYM
    ## INTJ CCONJ X NOUN DET
    ## PROPN NUM VERB PART PRON SCONJ
    def get_similar(self, in_tokens: Iterable[Tuple[str, Iterable[str]]], topn=10) \
            -> List[List[Tuple[str, str, float]]]:
        """Returns the word embeddings in semantic space that are nearest to
        each token.

        :param in_tokens: The tokens for which to get semantic neighbors and the
            POS tags to accept for each token. If no POS tags are specified,
            all POS tags are accepted. Ex:
            [('расследовать_VERB', ['VERB']), ('вопрос_NOUN', [])]
        :param topn: The maximum amount of neighbors to return for each token.
        :return: For each token, a list of semantic neighbors with associated
            POS tag and cosine similarity score.
        """
        if not self.check_loaded():
            return []

        similar_words = []
        for token, accepted_pos in in_tokens:
            # word2vec cannot predict oov terms
            if self.model_type is self.Model.word2vec and \
                    token not in self.model.wv.vocab:
                logging.error("Error: OOV token: {}".format(str(token)))
                similar_words.append([])
            else:
                similar_words.append(
                    [tuple(word[0].split("_")) + (word[-1],) for word in self.model.wv.most_similar(positive=[token], topn=topn)])
                if len(accepted_pos) < 1:
                    similar_words[-1] = [(word, tag, score) for word, tag, score
                                         in similar_words[-1]]
                else:
                    similar_words[-1] = [(word, tag, score) for word, tag, score
                                     in similar_words[-1] if tag in accepted_pos]
        return similar_words

    def predict_similar(self, word_pairs: Iterable[Iterable[Tuple[str, Iterable[str]]]],
                        topn: int = 10, verbose: bool = False) -> List:
        """Run get_similar on one or more pairs of tokens.

        :param word_pairs: A list of inputs for the get_similar function.
        :param topn: The maximum amount of neighbors to return for each token in
            each word pair.
        :param verbose: Print the semantic neighbors for each token.
        :return: A list of the results from get_similar for each word pair.
        """
        if not self.check_loaded():
            return []

        ret = []
        for words in word_pairs:
            similar_words = self.get_similar(in_tokens=words, topn=topn)
            word_list = [[word[0], similar] for word, similar in zip(words, similar_words)]
            ret.append(word_list)

            if verbose:
                print("\nwords: {}".format(words))
                for word in word_list:
                    print("{}:\n{}"
                          .format(word[0], "\n".join(["{:3}. {}"
                                                     .format(idx, similar_word)
                                                      for idx, similar_word in enumerate(word[1], start=1)])))
        return ret

    def get_similar_combined(self, in_tokens: List[str],
                             accepted_pos: Iterable[str], topn=10,
                             cos_type: str = "cossim") \
            -> List[Tuple[str, str, float]]:
        if not self.check_loaded():
            return []

        similar_words = []
        # word2vec cannot predict oov terms
        if self.model_type is self.Model.word2vec and \
                any([token not in self.model.wv.vocab for token in in_tokens]):
            logging.error("Error: OOV token: {}".format(str(in_tokens)))
        else:
            similar_words = []
            if cos_type == "cossim":
                similar_words = [tuple(word[0].split("_")) + (word[-1],) for word in
                                 self.model.wv.most_similar(positive=in_tokens, topn=topn)]
            elif cos_type == "cosmul":
                similar_words = [tuple(word[0].split("_")) + (word[-1],) for word in
                                 self.model.wv.most_similar_cosmul(positive=in_tokens, topn=topn)]
            else:
                logging.error("Error: Invalid cos_type, not 'cossim' or 'cosmul': {}".format(cos_type))
                return []

            if len(accepted_pos) < 1:
                similar_words = [(word, tag, score) for word, tag, score
                                     in similar_words]
            else:
                similar_words = [(word, tag, score) for word, tag, score
                                     in similar_words if tag in accepted_pos]
        return similar_words

    def predict_similar_combined(self, word_pairs: Iterable[Iterable[str]], token_pairs = None,
                                 accepted_pos: Iterable[str] = None, topn: int = 10,
                                 verbose: bool = False, cos_type: str = "cossim") -> List:
        if not self.check_loaded():
            return []
        if accepted_pos is None:
            accepted_pos = []

        ret = []
        for idx, words in enumerate(word_pairs):
            similar_words = []
            # word2vec cannot predict oov terms
            if self.model_type is self.Model.word2vec and \
                    any([token not in self.model.wv.vocab for token in words]):
                logging.error("Error: OOV token: {}".format(str(words)))
            else:
                if cos_type == "cossim":
                    similar_words = [tuple(word[0].split("_")) + (word[-1],) for word in
                                     self.model.wv.most_similar(positive=words, topn=topn)]
                elif cos_type == "cosmul":
                    similar_words = [tuple(word[0].split("_")) + (word[-1],) for word in
                                     self.model.wv.most_similar_cosmul(positive=words, topn=topn)]
                else:
                    logging.error("Error: Invalid cos_type, not 'cossim' or 'cosmul': {}".format(cos_type))
                    return []
            word_list = None
            if token_pairs is None:
                word_list = [words, similar_words]
            else:
                word_list = [token_pairs[idx], words, similar_words]
            ret.append(word_list)

            if verbose:
                print("\nwords: {}".format(words))
                if len(word_list[-1]) < 1:
                    print("No matches for", word_list)
                else:
                    print("{}:\n{}"
                          .format(word_list[-2], "\n".join(["{:3}. {}"
                                                           .format(idx, similar_word)
                                                            for idx, similar_word in enumerate(word_list[-1], start=1)])))
        return ret
