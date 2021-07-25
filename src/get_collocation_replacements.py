import configparser
import logging
import os
import pandas as pd

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


def getPosMask(n: int, replaceAmt: int):
    """Auxiliary function for StaticModels. Generates the PoS masks used for
    deciding which words in a given ngram for which suggested replacements
    will be found.

    :param n: 2 for bigrams, 3 for trigrams.
    :param replaceAmt: The amount of unfixed tokens.
        Ex. 1 for replacing only one word; 2 for replacing two words
    :return the PoS masks for the ngram.
    """
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


def getMasked(tokens: List[str], n: int, replaceAmt: int, pos: bool, mask: str):
    """Auxiliary function for DynamicModels. Masks replaceAmt tokens in the
    given input. The DynamicModel will then find suggested replacements for
    these masked tokens.

    :param tokens: A list of tokens or token_pos corresponding to the ngram.
        ex. ['прийти_v', 'к_p', 'вывод_n']
    :param n: 2 for bigrams, 3 for trigrams.
    :param replaceAmt: The amount of unfixed tokens.
        Ex. 1 for replacing only one word; 2 for replacing two words
    :param pos: Flag denoting the existence of PoS tags in the tokens.
        ex. 'прийти_v'
    :param mask: The mask token used when training the DynamicModel.
    :return a list of collocations with replaceAmt masks.
    """
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


def getCollocationComponents(collocations: List[str], n: int, pos: bool):
    """Auxiliary function for transforming a list of collocations into a format
    appropriate for the attest_collocations function.

    :param collocations: A list of collocations to break into components.
        Note: All collocations must comprise of the same number of tokens.
    :param n: 2 for bigrams, 3 for trigrams.
    :param pos: Flag denoting the existence of PoS tags in the tokens.
        ex. 'прийти_v'
    :return A list of n lists of strings, where each list is the unique set of
        tokens for the corresponding ngram position.
        ex. ["прийти к вывод", "прийти к заключение"] -> [["прийти"], ["к"] ,["вывод", "заключение"]]
    """
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


def getCollocationReplacements(collocations: List[str], staticModel: bool, modelType, modelSrc, binaryModel=True,
                                 cossim=True, mask="<mask>", topn=100, replace1=True, replace2=True, replace3=True,
                                 include_pmi=True, include_t_score=True, include_ngram_freq=True):
    """Given a list of collocations (lemmatized or not), this function will
    return a dictionary of attested suggested replacements and associated
    collocatiability statistics.

    :param collocations: A list of collocations for which replacements will be
        given.
    :param staticModel: True if using a StaticEmbedder, else false for DynamicEmbedder.
    :param modelType: The enum corresponding to the type of model to be used for
        suggesting collocation replacements. ex. StaticEmbedder.Model.GLOVE
    :param modelSrc: The path to the embedding model to use.
    :param binaryModel: Only relevant for GloVe embeddings. True if reading from
        a .model file, else False if reading from a .txt file.
    :param cossim: Only relevant for StaticEmbedders. The type of collocation
        metric to use. True for cossim, else false for cosmul.
    :param mask: Only relevant for DynamicEmbedders. The mask token used when
        training the model. Default: "<mask>"
    :param topn: The amount of collocations the embedding model should suggest.
        Note: This does not translate to the amount of collocation suggestions
        this function will return. Default: suggest 100 replacements
    :param replace1: Find replacements for up to 1 token in each collocation.
    :param replace2: Find replacements for up to 2 tokens in each collocation.
    :param replace3: Find replacements for up to 3 tokens in each collocation.
    :param include_pmi: Calculate the pmi score for attested collocation suggestions.
    :param include_t_score: Calculate the t-score for attested collocation suggestions.
    :param include_ngram_freq: Calculate the frequency for the attested collocation suggestions.
    :return a dictionary of attested collocation suggestions for each given collocation.
        Ex:
        {'исследовать_v вопрос_n':
            [{'suggested': 'анализировать вопрос', 'rank': 0, 'cosScore': 0.5687171816825867, 'numReplaced': 1, 'pmi': -1.9326116004215073, 't_score': -713.0812254264505, 'ngram_freq': 71},
            {'suggested': 'рассматривать вопрос', 'rank': 1, 'cosScore': 0.5575243234634399, 'numReplaced': 1, 'pmi': -0.8382014019729593, 't_score': -322.48555306741184, 'ngram_freq': 2998},
        'школа_n экономика_n':
            [{'suggested': 'учитель экономика', 'rank': 1, 'cosScore': 0.4992271065711975, 'numReplaced': 1, 'pmi': -1.9473737666819497, 't_score': -689.6667862412795, 'ngram_freq': 62},
            {'suggested': 'учитель экономика', 'rank': 1, 'cosScore': 0.4992271065711975, 'numReplaced': 2, 'pmi': -1.9473737666819497, 't_score': -689.6667862412795, 'ngram_freq': 62}]
        }

    """
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


def writeResults(collocation_dictionary, save_folder=""):
    """Write a attested collocation suggestions in the return format of the
    getCollocationReplacements to a specified folder. Files have the name of
    the input collocation, with the contents consisting of the suggested
    collocation and its associated statistics. Files are saved in csv format
    with ".txt" extension.

    :param collocation_dictionary: A dictionary of input collocations to
        attested collocation suggestions.
    :param save_folder: The folder where files will be saved.
    """
    if save_folder == "":
        save_folder = str(Path(log_dir))

    for colloc, suggest_dict in collocation_dictionary.items():
        save_path = str(Path(save_folder, colloc + ".txt"))
        df = pd.DataFrame(suggest_dict)
        df.to_csv(save_path, encoding="utf-8")
