from gensim import utils
from gensim.models import FastText, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from train_utils import SentencesLoader, LossLogger

import tempfile
import logging
import os
import configparser

path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, 'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
models_dir = config['PATHS']['models_dir']
data_dir = config['PATHS']['data_dir']
lemmas_dir = 'preprocessed/full_domains/lemmas/'
log_dir = config['PATHS']['log_dir']
log_file = os.path.join(log_dir, 'training.txt')
# logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
#                     format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

paths = [  
  # Lemmatized w/ treetagger
  os.path.join(data_dir, lemmas_dir, 'Economics.txt'),
  os.path.join(data_dir, lemmas_dir, 'Education_and_psychology.txt'),
  os.path.join(data_dir, lemmas_dir, 'History.txt'),
  os.path.join(data_dir, lemmas_dir, 'Law.txt'),
  os.path.join(data_dir, lemmas_dir, 'Linguistics.txt'),
  os.path.join(data_dir, lemmas_dir, 'Sociology.txt'),
  os.path.join(data_dir, lemmas_dir, 'supercybercat.txt'),

  # Lemmatized w/ UDPipe
  #  str(Path('./drive/MyDrive/Training_Data/CAT_sentences_full_lemma_1.txt')),
  #  str(Path('./drive/MyDrive/Training_Data/CAT_sentences_full_lemma_2.txt')),
  #  str(Path('./drive/MyDrive/Training_Data/cybercat_sentences_full_lemma.txt'))
]


CONTEXT_WINDOW = 5 # 5, 10
MIN_COUNT = 5
# EPOCHS = 100
SIZE = 500 # 200, 300, 500
CORES = 40

def main():
  for epochs in [30, 100]:
    for datapath in paths:
      print(f'Started training for: {datapath}', flush=True)
      model_name = os.path.splitext(os.path.basename(datapath))[0] + f'_{epochs}epx.model'
      sentences = SentencesLoader(datapath)
      w2v_loss_logger = LossLogger()
      w2v_model = Word2Vec(sentences=sentences, size=SIZE, window=CONTEXT_WINDOW, 
                          min_count=MIN_COUNT, workers=CORES, iter=epochs, 
                          callbacks=[w2v_loss_logger], compute_loss=True,)
      w2v_model.save(os.path.join(models_dir, 'w2v', model_name))
      print(w2v_loss_logger.losses, flush=True)



if __name__ == '__main__':
    main()