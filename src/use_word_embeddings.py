import configparser
import logging
import mysql.connector
import os

import math

import src.scripts.preprocessing.kutuzov.rus_preprocessing_udpipe as kutuzov

from pathlib import Path
from src.word_embedder import WordEmbedder
from typing import Iterable, List, Optional, Tuple
from src.scripts.colloc import pmi, t_score

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
models_dir = config['PATHS']['models_dir']
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'use_word_embeddings.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DOMAIN = config['SERVER']['DOMAIN']
HOST = config['SERVER']['HOST']
USER = config['SERVER']['USER']
PWD = config['SERVER']['PWD']

# Settings for using the word embeddings
USE_W2V = False
USE_FASTTEXT = True
PREDICT = False
PREDICT_COMBINED = True
USE_COSSIM = True
USE_COSMUL = True
ATTEST_COLLOCATIONS = True
WRITE_COLLOCATIONS = True

# Initialize word2vec and fastText word embedders
w2v_embedder = WordEmbedder(model_type=WordEmbedder.Model.word2vec)
fastText_embedder = WordEmbedder(model_type=WordEmbedder.Model.fastText)


def generate_word_pairs(sentences: Iterable[str]):
    word_pairs = []
    for sentence in sentences:
        word_pairs.append(kutuzov.process(kutuzov.process_pipeline, text=sentence,
                                          keep_pos=True, keep_punct=False))
    return word_pairs


def get_domain_size():
    connection = mysql.connector.connect(host=HOST, database=DOMAIN, user=USER, password=PWD)
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT COUNT(distinct lemma)
        FROM ruscorpora.lexicon;
        """)
    query_result = cursor.fetchone()
    cursor.close()
    connection.close()
    return query_result[0]


def get_unigram_freqs(lemmas: Iterable):
    connection = mysql.connector.connect(host=HOST, database=DOMAIN, user=USER, password=PWD)
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT lemma_1, SUM(ngram_freq)
        FROM ruscorpora.ngrams_1_lex_mv
        WHERE lemma_1 in ('{}')
        GROUP BY lemma_1;
        """.format("','".join(lemmas)))
    query_result = cursor.fetchall()
    cursor.close()
    connection.close()
    return query_result


def get_bigram_freqs(patterns: Iterable):
    connection = mysql.connector.connect(host=HOST, database=DOMAIN, user=USER, password=PWD)
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT CONCAT(lemma_1, " ", lemma_2), SUM(ngram_freq)
        FROM ruscorpora.ngrams_2_lex_mv
        WHERE CONCAT(lemma_1, " ", lemma_2) in ('{}')
        GROUP BY CONCAT(lemma_1, " ", lemma_2);
        """.format("','".join(patterns)))
    query_result = cursor.fetchall()
    cursor.close()
    connection.close()
    return query_result


def get_attested_collocations(result):
    tokens = result[-3]
    words = result[-2]
    word_lemma = " ".join([n.split("_")[0] for n in result[-2]])
    candidate_collocates = result[-1]
    pos_filtered_lists = []
    for word in words:  # ex. [расследовать_VERB, вопрос_NOUN, ...]
        word_pos = word.split("_")
        pos_filtered_lists.append([(lemma, score) for lemma, pos, score in candidate_collocates
                                   if pos in word_pos[1]])
        pos_filtered_lists[-1].insert(0, (word_pos[0], 1))
    attested_collocations = WordEmbedder.filter_results(
        [[n[0] for n in list]
           for list in pos_filtered_lists])

    ranked_attested_collocations = []
    # Amt of tokens in the db
    domain_size = get_domain_size()

    # dictionary for unigrams and bigrams
    bigrams = True
    if len(attested_collocations) > 0 and len(attested_collocations[0][:-2]) > 2:
        bigrams = False
    patterns = []
    last_words = []
    pattern_dict = {}
    last_word_dict = {}
    for collocate in attested_collocations:
        pattern = collocate[0]
        if not bigrams:
            pattern = " ".join(collocate[:2])
        last_word = collocate[-3]
        patterns.append(pattern)
        last_words.append(last_word)
        pattern_dict[pattern] = 0
        last_word_dict[last_word] = 0

    if bigrams:
        for row in get_unigram_freqs(patterns):
            pattern_dict[row[0]] = row[1]
    else:
        for row in get_bigram_freqs(patterns):
            pattern_dict[row[0]] = row[1]
    for row in get_unigram_freqs(last_words):
        last_word_dict[row[0]] = row[1]

    for collocate in attested_collocations:
        ngram_freq = collocate[-2]
        doc_freq = collocate[-1]
        colloc = " ".join(collocate[:-2])
        pattern = collocate[0]
        if not bigrams:
            pattern = " ".join(collocate[:2])
        last_word = collocate[-3]
        pmi_score = pmi(ngram_freq, pattern_dict[pattern], last_word_dict[last_word], domain_size)
        t_score_score = t_score(ngram_freq, pattern_dict[pattern], last_word_dict[last_word], domain_size)
        log_rank = 0
        sum_of_rank_rank = 0
        for rank_idx, list in enumerate(pos_filtered_lists):
            rank = [n[0] for n in pos_filtered_lists[rank_idx]].index(collocate[rank_idx])
            score = pos_filtered_lists[rank_idx][rank][-1]
            log_rank += rank
            sum_of_rank_rank += math.log(score)
        ranked_attested_collocations.append((tokens, word_lemma, colloc,
                                             log_rank, sum_of_rank_rank,
                                             ngram_freq, doc_freq,
                                             pmi_score, t_score_score))
    return sorted(ranked_attested_collocations, key=lambda x: x[-6])


def write_result(result, out_path, mode="w"):
    if not os.path.exists(Path(out_path).parent):
        os.makedirs(Path(out_path).parent)

    # Save result as a csv file
    with open(out_path, mode, encoding='utf8') as file:
        for collocation in result:
            file.write(",".join([str(colloc) for colloc in collocation]) + "\n")

def predict(model_type, model_path, word_pairs, token_pairs, cos_type="cossim", topn=100,
            verbose=False, attest_collocations=False, write_collocations=False,
            out_path="", include_header=True):
    embedder = None
    if model_type == "w2v":
        embedder = w2v_embedder
        if not embedder.load_model(model_path=model_path):
            logging.error("Failed to load model: {}".format(model_path))
            return
    elif model_type == "fastText":
        embedder = fastText_embedder
        if not embedder.load_model(model_path=model_path):
            logging.error("Failed to load model: {}".format(model_path))
            return
    else:
        logging.error("Error: Invalid model_type not 'w2v' or 'fastTest': {}".format(model_type))
        return

    print("Semantic neighbors using the {} model:".format(model_path))
    results = embedder.predict_similar_combined(word_pairs=word_pairs, topn=topn, token_pairs=token_pairs,
                                                verbose=verbose, cos_type=cos_type)
    if attest_collocations:
        for result in results:
            attested_collocations = get_attested_collocations(result)
            print("Results for:", result[1])
            for idx, collocation in enumerate(attested_collocations, start=1):
                print("{:3}. {}".format(idx, ", ".join([str(part) for part in collocation])))
            print("-" * 50, "\n")

            # Write results to csv file
            if write_collocations:
                if include_header:
                    attested_collocations.insert(0, ["token", "lemma", "suggested_collocation",
                                                     "sum_of_rank", "log_rank", "ngram_freq", "doc_freq",
                                                     "pmi", "t_score"])
                    write_result(attested_collocations, out_path + str(result[0]) + ".txt")
    return results


def main():
    # Use the trained models to predict word collocations
    if PREDICT:
        if USE_W2V:
            w2v_model = str(Path(models_dir, 'word2vec', 'all.model'))
            fastText_model = str(Path(models_dir, 'fastText', 'all.model'))
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
        if USE_FASTTEXT:
            if fastText_embedder.load_model(model_path=fastText_model):
                print("Semantic neighbors using the 'Law' fastText model:")
                fastText_embedder.predict_similar(word_pairs=word_pairs, topn=10, verbose=True)
            else:
                logging.error("Failed to load model: {}".format(fastText_model))

    # Use the trained models to predict word collocations
    if PREDICT_COMBINED:
        w2v_model = str(Path(models_dir, 'word2vec', 'all.model'))
        fastText_model = str(Path(models_dir, 'fastText', 'all.model'))
        token_pairs = [
            'расследовать вопрос',
            'большая важность',
            'по наш расчет',
            'сыграть важный роль'
        ]
        word_pairs = generate_word_pairs(token_pairs)

        # Example of prediction using word2vec and the model trained on the entire corpus
        if USE_W2V:
            if USE_COSSIM:
                out_file = "cossim_w2v_all_"
                out_path = str(Path(log_dir, "cossim", out_file))
                predict(model_type="w2v", model_path=w2v_model, word_pairs=word_pairs, token_pairs=token_pairs,
                        cos_type="cossim", topn=100, verbose=not ATTEST_COLLOCATIONS,
                        attest_collocations=ATTEST_COLLOCATIONS, include_header=True,
                        write_collocations=WRITE_COLLOCATIONS, out_path=out_path)
            if USE_COSMUL:
                out_file = "cosmul_w2v_all_"
                out_path = str(Path(log_dir, "cosmul", out_file))
                predict(model_type="w2v", model_path=w2v_model, word_pairs=word_pairs, token_pairs=token_pairs,
                        cos_type="cosmul", topn=100, verbose=not ATTEST_COLLOCATIONS,
                        attest_collocations=ATTEST_COLLOCATIONS, include_header=True,
                        write_collocations=WRITE_COLLOCATIONS, out_path=out_path)

            # if w2v_embedder.load_model(model_path=w2v_model):
            #     print("Semantic neighbors using the w2v model:")
            #     results = w2v_embedder.predict_similar_combined(word_pairs=word_pairs, topn=100, verbose=not ATTEST_COLLOCATIONS)
            #
            #     if ATTEST_COLLOCATIONS:
            #         for result_idx, result in enumerate(results):
            #             attested_collocations = get_attested_collocations(result)
            #             print("Results for:", word_pairs[result_idx])
            #             for idx, collocation in enumerate(attested_collocations, start=1):
            #                 print("{:3}. {}".format(idx, collocation))
            #             print("-"*50, "\n")
            #
            #             # Write results to csv file
            #             if WRITE_COLLOCATIONS:
            #                 out_file = "w2v_all_" + str(" ".join(result[0]))
            #                 out_path = str(Path(log_dir, out_file)) + ".csv"
            #                 write_result(attested_collocations, out_path)
            # else:
            #     logging.error("Failed to load model: {}".format(w2v_model))

        # Example of prediction using fastText and the model trained on the 'all' subcorpus
        if USE_FASTTEXT:
            if USE_COSSIM:
                out_file = "cossim_fastText_all_"
                out_path = str(Path(log_dir, "cossim", out_file))
                predict(model_type="fastText", model_path=fastText_model, word_pairs=word_pairs, token_pairs=token_pairs,
                        cos_type="cossim", topn=100, verbose=not ATTEST_COLLOCATIONS,
                        attest_collocations=ATTEST_COLLOCATIONS, include_header=True,
                        write_collocations=WRITE_COLLOCATIONS, out_path=out_path)
            if USE_COSMUL:
                out_file = "cosmul_fastText_all_"
                out_path = str(Path(log_dir, "cosmul", out_file))
                predict(model_type="fastText", model_path=fastText_model, word_pairs=word_pairs, token_pairs=token_pairs,
                        cos_type="cosmul", topn=100, verbose=not ATTEST_COLLOCATIONS,
                        attest_collocations=ATTEST_COLLOCATIONS, include_header=True,
                        write_collocations=WRITE_COLLOCATIONS, out_path=out_path)

            # if fastText_embedder.load_model(model_path=fastText_model):
            #     print("Semantic neighbors using the 'all' fastText model:")
            #     results = fastText_embedder.predict_similar_combined(word_pairs=word_pairs, topn=100,
            #                                                          verbose=not ATTEST_COLLOCATIONS, cos_type="cossim")
            #     if ATTEST_COLLOCATIONS:
            #         for result_idx, result in enumerate(results):
            #             attested_collocations = get_attested_collocations(result)
            #             print("Results for:", word_pairs[result_idx])
            #             for idx, collocation in enumerate(attested_collocations, start=1):
            #                 print("{:3}. {}".format(idx, collocation))
            #             print("-"*50, "\n")
            #
            #             # Write results to csv file
            #             if WRITE_COLLOCATIONS:
            #                 out_file = "fastText_all_" + str(" ".join(result[0]))
            #                 out_path = str(Path(log_dir, out_file)) + ".csv"
            #                 write_result(attested_collocations, out_path)
            # else:
            #     logging.error("Failed to load model: {}".format(fastText_model))


if __name__ == "__main__":
    main()
