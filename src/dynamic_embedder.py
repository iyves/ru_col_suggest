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
    """The class for suggesting collocation via dynamic word embeddings.
    """

    class Model(Enum):
        """ The type of dynamic word embeddings to use.
        """
        BERT = 1

    def __init__(self, model_type, src: str, fine_tuned: bool = False, tokenizer_name: str = None):
        """Initialize the embedding model.

        :param model_type: The type of embeddings to use.
        :param src: The path to the location of the word embedding model.
        :param fine_tuned: whether the model was fined-tuned from a pretrained preexisting model
        :param tokenizer_name: name of the tokenizer used during training
        """
        self.model_type = model_type
        self.fine_tuned = fine_tuned
        self.tokenizer_name = tokenizer_name
        if self.fine_tuned and not self.tokenizer_name:
            raise ValueError("If 'fine_tuned' is set to True, you should pass a huggingface tokenizer name to 'tokenizer_name'")
        self.load_model(src)

    def load_model(self, src: str):
        if self.model_type == self.Model.BERT:
            self.model = AutoModelForMaskedLM.from_pretrained(src)
            if self.fine_tuned:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(src)
        else:
            logging.error("Can only load bert models.")
            raise NotImplementedError

    def suggest_collocations(self, masked_ngram: str, topn: int = 10) -> List[Tuple[str, int]]:
        """Return a ranked list of collocations from a input sentence with masked tokens.

        :param masked_ngram: The input sentence with masked tokens.
            ex. "рассматривать_V <mask>_N экономика_N" (model trained w/ pos tags) or
                "рассматривать <mask> <mask>" (model trained w/o pos tags)
        :param topn: The amount of tokens to suggest for each masked token.
        :return a ranked list of suggested collocations.
            ex. [("рассматривать школа экономика", 3), ("рассматривать потребность экономика", 5)] (model trained w/ pos tags)or
            [("рассматривать школу экономики", 3), ("рассматривать потребность экономики", 5)] (model trained w/o pos tags)
        """
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
                if "mask" or "MASK" in token:
                    continue
                else:
                    suggested_replacements.insert(idx, [(0, token)])

            # Flatten out the suggested_replacements list to format (rank, colloc)
            suggested_collocations = []
            for colloc_tuple in product(*suggested_replacements):
                colloc = []
                rank = 0
                for tup in colloc_tuple:
                    colloc.append(tup[1])
                    rank += tup[0]
                suggested_collocations.append((" ".join(colloc), rank))
            return suggested_collocations
        else:
            logging.error("Can only load bert models.")
            raise NotImplementedError
