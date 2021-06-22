import configparser
import logging
import os

import pytest
import csv

from pathlib import Path
from src.helpers import get_text, get_file_names
from src.word_embedder import WordEmbedder
from src.use_word_embeddings import predict, generate_word_pairs, get_attested_collocations, write_result

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
models_dir = config['PATHS']['models_dir']

# Initialize word2vec and fastText word embedders
# w2v_model = str(Path(models_dir, 'w2v_colab', 'w2v.model'))
# w2v_embedder = WordEmbedder(model_type=WordEmbedder.Model.word2vec)
# w2v_embedder.load_model(model_path=w2v_model)

fastText_model = str(Path(models_dir, 'fastText_colab', 'fastText.model'))
fastText_embedder = WordEmbedder(model_type=WordEmbedder.Model.fastText)
fastText_embedder.load_model(model_path=fastText_model)

test_folder = Path("./fixtures/")
output_folder = Path("./fixtures/out")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


@pytest.fixture
def word_embedder(request) -> WordEmbedder:
    type = request.param
    return WordEmbedder(type)


# Read in collocations to evaluate from a csv file
evaluation_path = str(Path(test_folder, "Коллокации для эвалюации2.csv"))
evaluation_collocations = []
with open(evaluation_path, "r", encoding='utf8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header line
    for line in reader:
        evaluation_collocations.append(line)


class TestWordEmbedder:
    # @pytest.mark.parametrize('word_embedder, model_type', [
    #     (WordEmbedder.Model.word2vec, WordEmbedder.Model.word2vec),
    #     (WordEmbedder.Model.fastText, WordEmbedder.Model.fastText)
    # ], indirect=['word_embedder'])
    # def test_Initialize_ShouldHaveDefaultValue(
    #         self, word_embedder, model_type, capsys):
    #     assert word_embedder.model_type == model_type
    #     assert word_embedder.model is None
    #
    #     captured = capsys.readouterr()
    #     assert captured.out != ""
    #     assert captured.err == ""

    @pytest.mark.parametrize("collocation,lemma,t1,t2,t3,t4", evaluation_collocations)
    def test_TokenizeCollocations_ResultInLemma(self, collocation, lemma,
                                                t1, t2, t3, t4):
        token = generate_word_pairs([collocation])[0]
        token = " ".join([t.split("_")[0] for t in token])
        assert token == lemma

    @pytest.mark.parametrize("collocation,lemma,t1,t2,t3,t4", evaluation_collocations)
    def test_GetSuggestionsCossimW2V_ResultInLemma(self, collocation, lemma,
                                             t1, t2, t3, t4):
        token_pairs = [t1, t2, t3, t4]
        word_pairs = generate_word_pairs(token_pairs)

        results = w2v_embedder.predict_similar(
            word_pairs=word_pairs, topn=100, token_pairs=token_pairs,
            verbose=False, cos_type="cossim")

        all_attested_collocations = []
        for result in results:
            all_attested_collocations.append(result)
        all_attested_collocations.insert(0, [["num_replacements", "token", "lemma", "suggested_collocation",
                                              "log_rank", "sum_of_rank", "ngram_freq", "doc_freq",
                                              "pmi", "t_score"]])
        out_file = "cossim_w2v_all_" + collocation + ".txt"
        out_path = str(Path(output_folder, "w2v_colab", "cossim", out_file))
        flattened_list = [colloc for attest in all_attested_collocations for colloc in attest]
        write_result(flattened_list, out_path)

        # results = w2v_embedder.predict_similar_combined(word_pairs=word_pairs, topn=100,
        #                                                 verbose=False)
        # all_attested_collocations = []
        # for result_idx, result in enumerate(results):
        #     attested_collocations = get_attested_collocations(result)
        #     attested_collocations = [(pairs[result_idx], colloc, score)
        #                              for colloc, score in attested_collocations]
        #     all_attested_collocations.append(attested_collocations)
        # out_file = "w2v_all_" + lemma
        # out_path = str(Path(output_folder, out_file)) + ".csv"
        # flattened_list = [colloc for attest in all_attested_collocations for colloc in attest]
        # write_result(flattened_list, out_path)
        # for collocations in all_attested_collocations:
        #     assert lemma in collocations

    @pytest.mark.parametrize("collocation,lemma,t1,t2,t3,t4", evaluation_collocations)
    def test_GetSuggestionsCosmulW2V_ResultInLemma(self, collocation, lemma,
                                                   t1, t2, t3, t4):
        token_pairs = [t1, t2, t3, t4]
        word_pairs = generate_word_pairs(token_pairs)

        results = w2v_embedder.predict_similar(
            word_pairs=word_pairs, topn=100, token_pairs=token_pairs,
            verbose=False, cos_type="cosmul")

        all_attested_collocations = []
        for result in results:
            # attested_collocations = get_attested_collocations(result)
            all_attested_collocations.append(result)
        all_attested_collocations.insert(0, [["num_replacements", "token", "lemma", "suggested_collocation",
                                              "log_rank", "sum_of_rank", "ngram_freq", "doc_freq",
                                              "pmi", "t_score"]])

        out_file = "cosmul_w2v_all_" + collocation + ".txt"
        out_path = str(Path(output_folder, "w2v_colab", "cosmul", out_file))
        flattened_list = [colloc for attest in all_attested_collocations for colloc in attest]
        write_result(flattened_list, out_path)

    @pytest.mark.parametrize("collocation,lemma,t1,t2,t3,t4", evaluation_collocations)
    def test_GetSuggestionsCossimFasttText_ResultInLemma(self, collocation, lemma,
                                                   t1, t2, t3, t4):
        token_pairs = [t1, t2, t3, t4]
        word_pairs = generate_word_pairs(token_pairs)

        results = fastText_embedder.predict_similar(
            word_pairs=word_pairs, topn=100, token_pairs=token_pairs,
            verbose=False, cos_type="cossim")

        all_attested_collocations = []
        for result in results:
            # attested_collocations = get_attested_collocations(result)
            all_attested_collocations.append(result)
        all_attested_collocations.insert(0, [["num_replacements", "token", "lemma", "suggested_collocation",
                                              "log_rank", "sum_of_rank", "ngram_freq", "doc_freq",
                                              "pmi", "t_score"]])

        out_file = "cossim_fastText_all_" + collocation + ".txt"
        out_path = str(Path(output_folder, "fastText_colab", "cossim", out_file))
        flattened_list = [colloc for attest in all_attested_collocations for colloc in attest]
        write_result(flattened_list, out_path)

        # pairs = [t1, t2, t3, t4]
        # word_pairs = generate_word_pairs(pairs)
        # results = fastText_embedder.predict_similar_combined(word_pairs=word_pairs, topn=100,
        #                                                      verbose=False)
        # all_attested_collocations = []
        # for result_idx, result in enumerate(results):
        #     attested_collocations = get_attested_collocations(result)
        #     attested_collocations = [(pairs[result_idx], colloc, score)
        #                              for colloc, score in attested_collocations]
        #     all_attested_collocations.append(attested_collocations)
        # out_file = "fastText_all_" + lemma
        # out_path = str(Path(output_folder, out_file)) + ".csv"
        # flattened_list = [colloc for attest in all_attested_collocations for colloc in attest]
        # write_result(flattened_list, out_path)
        # for collocations in all_attested_collocations:
        #     assert lemma in collocations

    @pytest.mark.parametrize("collocation,lemma,t1,t2,t3,t4", evaluation_collocations)
    def test_GetSuggestionsCosmulFasttText_ResultInLemma(self, collocation, lemma,
                                                         t1, t2, t3, t4):
        token_pairs = [t1, t2, t3, t4]
        word_pairs = generate_word_pairs(token_pairs)

        results = fastText_embedder.predict_similar(
            word_pairs=word_pairs, topn=100, token_pairs=token_pairs,
            verbose=False, cos_type="cosmul")

        all_attested_collocations = []
        for result in results:
            # attested_collocations = get_attested_collocations(result)
            all_attested_collocations.append(result)
        all_attested_collocations.insert(0, [["num_replacements", "token", "lemma", "suggested_collocation",
                                              "log_rank", "sum_of_rank", "ngram_freq", "doc_freq",
                                              "pmi", "t_score"]])

        out_file = "cosmul_fastText_all_" + collocation + ".txt"
        out_path = str(Path(output_folder, "fastText_colab", "cosmul", out_file))
        flattened_list = [colloc for attest in all_attested_collocations for colloc in attest]
        write_result(flattened_list, out_path)
