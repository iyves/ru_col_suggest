import codecs
import configparser
import logging
import os
import pandas
import treetaggerwrapper
from typing import Iterable, List

from enum import Enum
from pathlib import Path

import src.scripts.preprocessing.kutuzov.rus_preprocessing_udpipe as kutuzov

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
treetagger_dir = config['PATHS']['treetagger_dir']
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'tokenize.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Tokenizer:
    """Process Russian text into lemmatized tokens.
    """

    accepted_tags = ['A', 'C', 'I', 'M', 'N', 'P', 'Q', 'R', 'S', 'V']
    tagger = None
    tag_table = None
    process_pipeline = None

    class Method(Enum):
        """The method to use for tokenization.
        TREETAGGER - Default.
        UDPIPE - Method used by Kutuzov.
        MYSTEM - Method developed by Yandex.
        """
        TREETAGGER = 1
        UDPIPE = 2
        MYSTEM = 3

    def __init__(self, method=Method.TREETAGGER):
        """Sets the method to use for tokenization
        :param method: The method to use for tokenization, default: TREETAGGER
        """
        self.method = method
        if self.method == self.Method.TREETAGGER:
            if Tokenizer.tagger is None:
                Tokenizer.tagger = treetaggerwrapper.TreeTagger(TAGLANG='ru', TAGDIR=str(Path(treetagger_dir)))
            if Tokenizer.tag_table is None:
                Tokenizer.tag_table = pandas.read_csv(str(Path(treetagger_dir, 'ru-table.tab')), sep='\t')
        elif self.method == self.Method.UDPIPE:
            if Tokenizer.process_pipeline is None:
                Tokenizer.model, Tokenizer.process_pipeline = kutuzov.load_model()
        else:
            logging.error("Can only tokenize via treetagger, udpipe, or mystem")
            raise

    def tokenize(self, sentence: str, keep_pos: bool = True,
                 keep_punct: bool = False) -> List[str]:
        """Tokenize a list of sentences via the selected method.

        :param sentences: The input to preprocess.
        :param keep_pos: Flag for keeping the Parts-Of-Speech tag with the
            lemmatized token. Ex. банк_N. Default: True
        :param keep_punct: Flag for keeping punctuation tokens. Only relevant
            for UDPipe. Default: False
        :return A list containing the tokenized input sentences.
        """
        res = kutuzov.unify_sym(sentence.strip())
        if self.method is self.Method.UDPIPE:
            output = " ".join(kutuzov.process(Tokenizer.process_pipeline,
                                                text=res, keep_pos=keep_pos, keep_punct=keep_punct))
            output = output.replace("numarab_PROPN", "numarab_NUM").replace("numlat_PROPN", "numlat_NUM")
        elif self.method is self.Method.TREETAGGER:
            if res[-1] not in ['.', '!', '?']:
                res += '.'

            tags = Tokenizer.tagger.tag_text(res)
            tags = [t.split() for t in tags]
            sentence = []
            for t in tags:
                if keep_pos:
                    if t[1] != 'SENT' and t[1][0] in Tokenizer.accepted_tags:
                        lemma_tag = t[2] + "_" + t[1][0]
                        if t[0] == "numarab" or t[0] == "numlat":
                            lemma_tag = t[2] + "_M"
                        if keep_pos:
                            sentence.append(lemma_tag)
                else:
                    sentence.append(t[2])
            output = " ".join(sentence)
        elif self.method is self.Method.MYSTEM:
            raise NotImplementedError
        else:
            logging.error("Can only tokenize via treetagger, udpipe, or mystem")
            raise NotImplementedError
        return output

    def tokenize_file(self, input_file, output_file, keep_pos: bool = True,
                      keep_punct: bool = False, batch: int = 10000, start: int = 0,
                      end: int = -1) -> int:
        """Lemmatize and tokenize the text by paragraph with the specified method.

        :param input_file: The full path to the file that contains the text to
            tokenize. File should have the format of one sentence per new line.
        :param output_file: The full path to the file to write the tokenized text.
        :param keep_pos: Flag for keeping the Parts-Of-Speech tag with the
            lemmatized token. Ex. банк_N. Default: True
        :param keep_punct: Flag for keeping punctuation tokens. Only relevant
            for UDPipe. Default: False
        :param batch: The amount of sentences to tokenize before saving to file.
            Default: 10000
        :param start: When to begin lemmatization. Defaut: 0
        :param end: When to end lemmatization. Default: -1 (lemmatize everything)
        :return The amount of lines processed, as an int
        """
        total = 0
        sentences = []
        for sentence in codecs.open(input_file, "r", "utf-8"):
            total += 1
            if total > end > 0:
                break

            if total >= start:
                sentences.append(self.tokenize(sentence, keep_pos, keep_punct))

                if total % batch == 0:
                    logging.info(f"Processed {len(sentences)} sentences")
                    with codecs.open(output_file, 'a', 'utf-8') as to_write:
                        to_write.write("\n".join(sentences) + "\n")
                    sentences = []
        if total % batch != 0:
            with codecs.open(output_file, 'a', 'utf-8') as to_write:
                logging.info(f"Processed {len(sentences)} sentences")
                to_write.write("\n".join(sentences) + "\n")
        return total
