import configparser
import logging
import math
import os
import sys
import torch

from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from src.static_embedder import StaticEmbedder
from src.dynamic_embedder import DynamicEmbedder
from src.collocation_attestor import CollocationAttestor


# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'get_collocation_replacements.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DOMAIN = config['SERVER']['DOMAIN']
HOST = config['SERVER']['HOST']
USER = config['SERVER']['USER']
PWD = config['SERVER']['PWD']


def getPosMask(n, replaceAmt):
    if replaceAmt == 1:
        if n == 2:
            return [[0, 1], [1, 0]]
        else:
            return [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    elif replaceAmt == 2:
        if n == 2:
            return [[0, 0]]
        else:
            return [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    elif replaceAmt == 3:
        if n == 2:
            return [[0, 0]]
        else:
            return [[0, 0, 0]]
    else:
        return []


def getMasked(tokens, n, replaceAmt, pos, mask):
    ret = []
    if pos:
        token_pos = [token.split("_")[1] if len(token.split("_")) == 2 else "n"
                     for token in tokens]

    if replaceAmt == 1:
        if n == 2:
            if pos:
                ret.append(" ".join([mask + "_" + token_pos[0], tokens[1]]))
                ret.append(" ".join([tokens[0], mask + "_" + token_pos[1]]))
            else:
                ret.append(" ".join([mask, tokens[1]]))
                ret.append(" ".join([tokens[0], mask]))
        else:
            if pos:
                ret.append(" ".join([mask + "_" + token_pos[0], tokens[1], tokens[2]]))
                ret.append(" ".join([tokens[0], mask + "_" + token_pos[1], tokens[2]]))
                ret.append(" ".join([tokens[0], tokens[1], mask + "_" + token_pos[2]]))
            else:
                ret.append(" ".join([mask, tokens[1], tokens[2]]))
                ret.append(" ".join([tokens[0], mask, tokens[2]]))
                ret.append(" ".join([tokens[0], tokens[1], mask]))
    elif replaceAmt == 2:
        if n == 2:
            if pos:
                ret.append(" ".join([mask + "_" + token_pos[0], mask + "_" + token_pos[1]]))
            else:
                ret.append(" ".join([mask, mask]))
        else:
            if pos:
                ret.append(" ".join([tokens[0], mask + "_" + token_pos[1], mask + "_" + token_pos[2]]))
                ret.append(" ".join([mask + "_" + token_pos[0], tokens[1], mask + "_" + token_pos[2]]))
                ret.append(" ".join([mask + "_" + token_pos[0], mask + "_" + token_pos[1], tokens[2]]))
            else:
                ret.append(" ".join([tokens[0], mask, mask]))
                ret.append(" ".join([mask, tokens[1], mask]))
                ret.append(" ".join([mask, mask, tokens[2]]))
    elif replaceAmt == 3:
        if n == 2:
            if pos:
                ret.append(" ".join([mask + "_" + token_pos[0], mask + "_" + token_pos[1]]))
            else:
                ret.append(" ".join([mask, mask]))
        else:
            if pos:
                ret.append(" ".join([mask + "_" + token_pos[0], mask + "_" + token_pos[1], mask + "_" + token_pos[2]]))
            else:
                ret.append(" ".join([mask, mask, mask]))
    return ret


def getCollocationComponents(collocations, n, pos):
    components = []
    if n == 2:
        components.extend([set(), set()])
        for colloc in collocations:
            tokens = [c.split("_")[0] for c in colloc.split(" ")]
            components[0].add(tokens[0])
            components[1].add(tokens[1])
    elif n == 3:
        components.extend([set(), set(), set()])
        for colloc in collocations:
            tokens = [c.split("_")[0] for c in colloc.split(" ")] if pos else colloc.split(" ")
            components[0].add(tokens[0])
            components[1].add(tokens[1])
            components[2].add(tokens[2])
    return [list(c) for c in components]


def get_collocation_replacements(collocations: List[str], staticModel, modelType, modelSrc, binaryModel=True,
                                 cossim=True, mask="<mask>", topn=100, replace1=True, replace2=True, replace3=True,
                                 include_pmi=True, include_t_score=True, include_ngram_freq=True):
    # Initialize the dictionary to return in format dict[original colloc][new colloc][stat]
    colloc_dict = {colloc: [] for colloc in collocations}

    if topn < 1:
        logging.error("topn must be at least 1")
        return colloc_dict

    if not replace1 and not replace2 and not replace3:
        logging.error("Must replace at least 1 token")
        return colloc_dict

    # Can only run algorithm for bigrams or trigrams
    if len(collocations) < 1:
        return colloc_dict
    n = len(collocations[0].split(" "))
    if n != 2 and n != 3:
        logging.error("Can only get collocation suggestions for bigrams and trigrams")
        return colloc_dict

    # Determine if using PoS tags or not (see if the first token of the first collocation is in the format "token_pos")
    pos = True if len(collocations[0].split(" ")[0].split("_")) == 2 else False
    if staticModel and not pos:
        logging.error("Input for staticModels must contain PoS tags")
        return colloc_dict

    # Load the static or dynamic model
    embedder = None
    if staticModel:
        embedder = StaticEmbedder(modelType, modelSrc, binaryModel)
    else:
        embedder = DynamicEmbedder(modelType, modelSrc)
    if embedder is None:
        logging.error(f"Failed to load modelType: '{modelType}'")
        return colloc_dict

    for colloc in colloc_dict:
        # Get the collocation suggestions
        tokens = colloc.split(" ")
        for numReplace, replace in enumerate([replace1, replace2, replace3], start=1):
            if replace:
                if staticModel:
                    for pos in getPosMask(n, 1):
                        suggested = embedder.suggest_collocations(tokens, pos, topn=topn, cossim=cossim)

                        # Get stats for attested collocations
                        with CollocationAttestor() as attestor:
                            attested = attestor.attest_collocations(getCollocationComponents([s[0] for s in suggested], n, True))
                            if include_t_score or include_pmi or include_ngram_freq:
                                stats = attestor.get_collocation_stats(attested, include_pmi, include_t_score, include_ngram_freq)

                        # Add results to the dictionary to return
                        for s in suggested:
                            if s[0] in attested:
                                info = {"suggested": s[0],
                                        "rank": s[1],
                                        "cosScore": s[2],
                                        "numReplaced": numReplace
                                        }
                                if include_pmi:
                                    info["pmi"] = stats[s[0]]["pmi"]
                                if include_t_score:
                                    info["t_score"] = stats[s[0]]["t_score"]
                                if include_ngram_freq:
                                    info["ngram_freq"] = stats[s[0]]["ngram_freq"]
                                colloc_dict[colloc].append(info)
                else:
                    for masked in getMasked(tokens, n, 1, pos, mask):
                        suggested = embedder.suggest_collocations(masked, topn)

                        # Get stats for attested collocations
                        with CollocationAttestor() as attestor:
                            attested = attestor.attest_collocations(
                                getCollocationComponents([s[0] for s in suggested], n, pos))
                            if include_t_score or include_pmi or include_ngram_freq:
                                stats = attestor.get_collocation_stats(attested, include_pmi, include_t_score, include_ngram_freq)

                            # Add results to the dictionary to return
                            for s in suggested:
                                if s[0] in attested:
                                    info = {"suggested": s[0],
                                            "rank": s[1],
                                            "numReplaced": numReplace
                                            }
                                    if include_pmi:
                                        info["pmi"] = stats[s[0]]["pmi"]
                                    if include_t_score:
                                        info["t_score"] = stats[s[0]]["t_score"]
                                    if include_ngram_freq:
                                        info["ngram_freq"] = stats[s[0]]["ngram_freq"]
                                    colloc_dict[colloc].append(info)
    return colloc_dict