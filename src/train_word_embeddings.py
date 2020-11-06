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
TRAIN_CORPUS = False
TRAIN_SUBCORPORA = False
PREDICT = True

# Initialize word2vec and fastText word embedders
w2v_embedder = WordEmbedder(model_type=WordEmbedder.Model.word2vec)
fastText_embedder = WordEmbedder(model_type=WordEmbedder.Model.fastText)

# The six fields of research
fields = 'Economics Education_and_psychology History Law Linguistics Sociology'.split()


# Train the entire corpus
if TRAIN_CORPUS:
    print("Training the word2vec model on the entire corpus")
    if not w2v_embedder.train_corpus(pickles_dir=pickles_dir, source_folders=fields,
                                     target_dir=str(Path(models_dir, 'word2vec')),
                                     target_filename="all.model"):
        logging.error("Failed to train the word2vec model on the entire corpus")

    print("Training the fastText model on the entire corpus")
    if not fastText_embedder.train_corpus(pickles_dir=pickles_dir, source_folders=fields,
                                          target_dir=str(Path(models_dir, 'fastText')),
                                          target_filename="all.model"):
        logging.error("Failed to train the fastText model on the entire corpus")

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


# Use the trained models to predict word collocations
if PREDICT:
    w2v_model = str(Path(models_dir, 'word2vec', 'all.model'))
    fastText_model = str(Path(models_dir, 'fastText', 'Law.model'))
    word_pairs = [[('расследовать_VERB', ['VERB']), ('вопрос_NOUN', ['NOUN'])],
                  [('расслвать_VERB', ['VERB']), ('врос_NOUN', [])],
                  [('большой_ADJ', ['ADJ', 'ADP', 'ADV']), ('важность_NOUN', ['PROPN', 'NOUN', 'PRON'])]
                  ]
    # Example of prediction using word2vec and the model trained on the entire corpus
    if w2v_embedder.load_model(model_path=w2v_model):
        print("Semantic neighbors using the w2v model:")
        w2v_embedder.predict_similar(word_pairs=word_pairs, verbose=True)
    else:
        logging.error("Failed to load model: {}".format(w2v_model))

    # Example of prediction using fastText and the model trained on the 'Law' subcorpus
    if fastText_embedder.load_model(model_path=fastText_model):
        print("Semantic neighbors using the 'Law' fastText model:")
        fastText_embedder.predict_similar(word_pairs=word_pairs, topn=5, verbose=True)
    else:
        logging.error("Failed to load model: {}".format(fastText_model))
