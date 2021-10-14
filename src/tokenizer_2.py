import treetaggerwrapper
import os
import codecs
import configparser
import logging
from pathlib import Path
from typing import Iterable, List
from joblib import delayed, Parallel

import scripts.preprocessing.kutuzov.rus_preprocessing_udpipe as kutuzov

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
data_dir = config['PATHS']['data_dir']
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'tokenize.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

treetagger_dir = config['PATHS']['treetagger_dir']

tokenizer = treetaggerwrapper.TreeTagger(TAGLANG='ru', TAGDIR=str(Path(treetagger_dir)))
accepted_tags = ['A', 'C', 'I', 'M', 'N', 'P', 'Q', 'R', 'S', 'V']

def tokenize(sentence: str, keep_pos: bool = True,
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
        if res[-1] not in ['.', '!', '?']:
            res += '.'

        tags = tokenizer.tag_text(res)
        tags = [t.split() for t in tags]
        lemmas_sentence = []
        tokens_sentence = []
        
        for t in tags:
            if keep_pos:
                if t[1] != 'SENT' and t[1][0] in accepted_tags:
                    lemma_tag = t[2] + "_" + t[1][0]
                    token_tag = t[0] + "_" + t[1][0]
                    if t[0] == "numarab" or t[0] == "numlat":
                        token_tag = t[0] + "_M"
                        lemma_tag = t[2] + "_M"
                    if keep_pos:
                        lemmas_sentence.append(lemma_tag)
                        tokens_sentence.append(token_tag)
            else:
                lemmas_sentence.append(t[2])
                tokens_sentence.append(t[0])

        lemmas_output = " ".join(lemmas_sentence)
        tokens_output = " ".join(tokens_sentence)
        return lemmas_output, tokens_output

def tokenize_file(input_file, output_tokens_file, output_lemmas_file, keep_pos: bool = True, 
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
    tokens_sentences = []
    lemmas_sentences = []
    for sentence in codecs.open(input_file, "r", "utf-8"):
        total += 1
        if total > end > 0:
            break

        

        if total >= start:
            lemmas, tokens = tokenize(sentence, keep_pos, keep_punct)
            lemmas_sentences.append(lemmas)
            tokens_sentences.append(tokens)

            if total % batch == 0:
                logging.info(f"Processed {len(lemmas_sentences)} sentences")
                with codecs.open(output_lemmas_file, 'a', 'utf-8') as to_write:
                    to_write.write("\n".join(lemmas_sentences) + "\n")
                with codecs.open(output_tokens_file, 'a', 'utf-8') as to_write:
                    to_write.write("\n".join(tokens_sentences) + "\n")
                lemmas_sentences = []
                tokens_sentences = []
    if total % batch != 0:
        with codecs.open(output_lemmas_file, 'a', 'utf-8') as to_write:
            logging.info(f"Processed {len(lemmas_sentences)} sentences")
            to_write.write("\n".join(lemmas_sentences) + "\n")
        with codecs.open(output_tokens_file, 'a', 'utf-8') as to_write:
            logging.info(f"Processed {len(tokens_sentences)} sentences")
            to_write.write("\n".join(tokens_sentences) + "\n")
    return total


def tokenize_one(source_dir, target_tokens_dir, target_lemmas_dir, file):
    """
    Tags text file from source_dir to target_dir. 
    
    :param file: text file to tag divided into line breaks
    """
    filename = os.path.splitext(os.path.basename(file))[0]
    logging.info("Tagging text from file: {}".format(filename))
    print("\n\n", file)
    tokenize_file(os.path.join(source_dir, file + '.txt'),
                  os.path.join(target_tokens_dir, file + '.txt'),
                  os.path.join(target_lemmas_dir, file + '.txt')
                  )
    logging.info(f"Tagged file: {filename}")
    print(filename, end=', ')
    return True



source_dir = os.path.join(data_dir, 'preprocessed/full_domains/text')
target_tokens_dir = os.path.join(data_dir, 'preprocessed/full_domains/tokens')
target_lemmas_dir = os.path.join(data_dir, 'preprocessed/full_domains/lemmas')
# print(target_lemmas_dir)
for target in [target_tokens_dir, target_lemmas_dir]:
    if not os.path.exists(target):
        os.makedirs(target)

files = 'Economics Law'

for file in files.split():
    tokenize_one(source_dir, target_tokens_dir, target_lemmas_dir, file)
# element_information = Parallel(n_jobs=-1, backend='threading')(
#     delayed(tokenize_one)(source_dir, target_tokens_dir, target_lemmas_dir, file) for file in files)
print("Done!")
