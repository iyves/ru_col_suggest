import configparser
import logging
import math
import os
import sys
import torch

from enum import Enum
from itertools import product
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from typing import Iterable, List, Optional, Tuple


# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'dynamic_embedder.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DOMAIN = config['SERVER']['DOMAIN']
HOST = config['SERVER']['HOST']
USER = config['SERVER']['USER']
PWD = config['SERVER']['PWD']


class DynamicEmbedder():
    class Model(Enum):
        BERT = 1

    def __init__(self, model_type, src: str):
        self.model_type = model_type
        self.model = self.load_model(src)

    def load_model(self, src: str):
        if self.model_type == self.Model.BERT:
            self.model = AutoModelForMaskedLM.from_pretrained(src)
            self.tokenizer = AutoTokenizer.from_pretrained(src)
        else:
            logging.error("Can only load bert models.")
            raise NotImplementedError

    def suggest_collocations(self, masked_ngram: str, topn: int = 10) -> List[Tuple[str, int]]:
        # https://discuss.huggingface.co/t/having-multiple-mask-tokens-in-a-sentence/3493
        if self.model_type == self.Model.BERT:
            token_ids = self.tokenizer.encode(masked_ngram, return_tensors='pt')
            token_ids_tk = self.tokenizer.tokenize(masked_ngram, return_tensors='pt')

            masked_position = (token_ids.squeeze() == self.tokenizer.mask_token_id).nonzero()
            masked_pos = [mask.item() for mask in masked_position]

            with torch.no_grad():
                output = self.model(token_ids)

            last_hidden_state = output[0].squeeze()

            # Get the suggested mask replacements in format (rank, token)
            suggested_replacements = []
            for mask_index in masked_pos:
                mask_hidden_state = last_hidden_state[mask_index]
                idx = torch.topk(mask_hidden_state, k=topn, dim=0)[1]
                words = [self.tokenizer.decode(i.item()).strip() for i in idx]
                suggested_replacements.append(enumerate(words))
            for idx, token in enumerate(masked_ngram.split(" ")):
                token = token.split("_")[0]
                if "mask" in token:
                    continue
                else:
                    suggested_replacements.insert(idx, tuple(0, token))

            # Flatten out the suggested_replacements list
            suggested_collocations = []
            for colloc_tuple in product(*suggested_replacements):
                colloc = []
                rank = 0
                for tuple in colloc_tuple:
                    colloc.append(tuple[0])
                    rank += tuple[1]
                suggested_collocations.append((" ".join(colloc), rank))
            return suggested_collocations
        else:
            logging.error("Can only load bert models.")
            raise NotImplementedError
