import configparser
import logging
import os
from pathlib import Path

from src.word_embedder import WordEmbedder

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
models_dir = config['PATHS']['models_dir']
pickles_dir = config['PATHS']['pickles_dir']
preprocessed_text_dir = config['PATHS']['preprocessed_text_dir']
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'train_word_embeddings.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Settings for training
TRAIN_CORPUS = True
TRAIN_SUBCORPORA = False


def main():
    # Initialize word2vec and fastText word embedders
    w2v_embedder = WordEmbedder(model_type=WordEmbedder.Model.word2vec)
    fastText_embedder = WordEmbedder(model_type=WordEmbedder.Model.fastText)

    # The six fields of research
    fields = 'Economics Education_and_psychology History Law Linguistics Sociology'.split()


    # Train the entire corpus
    if TRAIN_CORPUS:
        print("Training the word2vec model on the entire corpus")
        params = {
            "size": 100,
            "window": 8,
            "min_count": 5
        }
        if not w2v_embedder.train_corpus(pickles_dir=pickles_dir, source_folders=fields,
                                         target_dir=str(Path(models_dir, 'word2vec')),
                                         target_filename="all.model", params=params):
            logging.error("Failed to train the word2vec model on the entire corpus")

        print("Training the fastText model on the entire corpus")
        if not fastText_embedder.train_corpus(pickles_dir=pickles_dir, source_folders=fields,
                                              target_dir=str(Path(models_dir, 'fastText')),
                                              target_filename="all.model"):
            logging.error("Failed to train the fastText model on the entire corpus", params=params)

    # Train each subcorpus
    if TRAIN_SUBCORPORA:
        for field in fields:
            print("Training the word2vec model on the {} subcorpus".format(field))
            if not w2v_embedder.train_subcorpora(pickles_dir=pickles_dir, source_folders=[field],
                                                 target_dir=str(Path(models_dir, 'word2vec')),
                                                 target_filenames="{}.model".format(field)):
                logging.error("Failed to train the word2vec model on the {} subcorpus".format(field))

            print("Training the fastText model on the {} subcorpus".format(field))
            if not fastText_embedder.train_subcorpora(pickles_dir=pickles_dir, source_folders=[field],
                                                      target_dir=str(Path(models_dir, 'fastText')),
                                                      target_filenames="{}.model".format(field)):
                logging.error("Failed to train the fastText model on the {} subcorpus".format(field))


if __name__ == "__main__":
    main()
