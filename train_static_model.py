from gensim import utils
from gensim.models import FastText, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from train_utils import SentencesLoader, LossLogger

import tempfile
import logging
import os
import configparser
import argparse
from enum import Enum

class Model(Enum):
  W2V = 1
  FASTTEXT = 2

def main():
  from argparse import ArgumentParser

  parser = ArgumentParser()

  parser.add_argument('model_name', type=str, help='Set to "w2v" to train a word2vec model, "fasttext" for a fastText model.')
  parser.add_argument('data_dir', type=str, help='Path to directory containing the text files for the dataset.')
  parser.add_argument('save_dir', type=str, help='Path to directory to save the models on.')
  parser.add_argument('--domains', type=str, 
                    help='Comma-separated (without blanks) list of the names of the files used for training (without extension): "Economics,Education_and_Psychology,History,Law,Linguistics,Sociology,supercybercat". They should be files in txt format with one sentence per line.')
  parser.add_argument('--context_window', type=int, help='Context window for Skip-gram model, usually set to 5 or 10.')
  parser.add_argument('--min_count', type=int, help='Ignores all words with total frequency lower than this.')
  parser.add_argument('--epochs', type=int, help='Number of training epochs.')
  parser.add_argument('--size' , type=int, help='Dimensionality of the word vector.')
  parser.add_argument('--workers', type=int, help='Use these many worker threads to train the model (=faster training with multicore machines).')

  parser.set_defaults(
    domains='supercybercat',
    context_windows=5,
    min_count=5,
    epochs=30,
    size=300,
    workers=40
  )

  args = parser.parse_args()

  for d in args.domains.split(','):
    file = os.path.join(data_dir, f'{d}.txt')
    if os.path.isfile(file):
      print(f'Started training for: {os.path.join(data_dir)}', flush=True)
      model_name = os.path.splitext(os.path.basename(file))[0] + f'_{args.size}vector_size.model'
      sentences = SentencesLoader(datapath)
      model = Model[args.model.upper()]
      if model == Model.W2V:
        w2v_loss_logger = LossLogger()
        w2v_model = Word2Vec(sentences=sentences, vector_size=args.size, window=args.window, 
                            min_count=args.min_count, workers=args.workers, epochs=args.epochs, 
                            callbacks=[w2v_loss_logger], compute_loss=True)
        save_to = os.path.join(save_dir, 'w2v', model_name)
        os.makedirs(save_to, exist_ok=True)
        w2v_model.save(save_to)
      elif model == Model.FASTTEXT:
        fastText_model = FastText(sentences=sentences, vector_size=args.size, window=args.window, 
                            min_count=args.min_count, workers=args.workers, epochs=args.epochs)
        save_to = os.path.join(save_dir, 'fastText', model_name)
        os.makedirs(save_to, exist_ok=True)
        fastText_model.save(save_to)
      else:
        print("The model you chose is not supported.")
      print(w2v_loss_logger.losses, flush=True)



if __name__ == '__main__':
    main()