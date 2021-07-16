import configparser
import os
import pytest

from src.tokenizer import Tokenizer

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)


@pytest.fixture
def tokenizer(request) -> Tokenizer:
    model_type = request.param
    return Tokenizer(model_type)


class TestTokenizer:
    @pytest.mark.parametrize('tokenizer, model_type', [
        (Tokenizer.Method.TREETAGGER, Tokenizer.Method.TREETAGGER),
    ], indirect=['tokenizer'])
    def test_InitTreeTagger_ShouldPass(self, tokenizer, model_type):
        assert tokenizer.method == model_type
        assert tokenizer.tagger is not None
        assert tokenizer.tag_table is not None

    @pytest.mark.parametrize('tokenizer, model_type', [
        (Tokenizer.Method.UDPIPE, Tokenizer.Method.UDPIPE)
    ], indirect=['tokenizer'])
    def test_InitUdpipe_ShouldPass(self, tokenizer, model_type):
        assert tokenizer.method == model_type
        assert tokenizer.process_pipeline is not None

    # Might fail if there is not enough memory to store the UDPipe model
    @pytest.mark.parametrize('tokenizer, input, expected_output', [
        (Tokenizer.Method.UDPIPE, [], []),
        (Tokenizer.Method.UDPIPE, ['по нашим подсчетам'], ['по_ADP наш_DET подсчет_NOUN']),
        (Tokenizer.Method.UDPIPE, ['по нашим подсчетам','в одинаковый вид','в конце мы сделали следующие выводы'],
         ['по_ADP наш_DET подсчет_NOUN','в_ADP одинаковый_ADJ вид_NOUN','в_ADP конец_NOUN мы_PRON делать_VERB следующий_ADJ вывод_NOUN'])
    ], indirect=['tokenizer'])
    def test_UdpipeTokenizeSentences_ShouldReturnSentences(self, tokenizer, input, expected_output):
        assert tokenizer.method == Tokenizer.Method.UDPIPE
        output = tokenizer.tokenize(input)
        assert len(output) == len(expected_output)
        for actual, expected in zip(output, expected_output):
            assert actual == expected

    @pytest.mark.parametrize('tokenizer, input, expected_output', [
        (Tokenizer.Method.TREETAGGER, [], []),
        (Tokenizer.Method.TREETAGGER, ['По нашим подсчетам'], ['по_S наш_P подсчет_N']),
        (Tokenizer.Method.TREETAGGER, ['По нашим подсчетам','в одинаковый вид','в конце мы сделали следующие выводы'],
         ['по_S наш_P подсчет_N', 'в_S одинаковый_A вид_N', 'в_S конец_N мы_P сделать_V следующий_A вывод_N'])
    ], indirect=['tokenizer'])
    def test_TreetaggerTokenizeSentences_ShouldReturnSentences(self, tokenizer, input, expected_output):
        assert tokenizer.method == Tokenizer.Method.TREETAGGER
        output = tokenizer.tokenize(input)
        assert len(output) == len(expected_output)
        for actual, expected in zip(output, expected_output):
            assert actual == expected

# 	- same, but don't keep pos ags
# 	- same, but keep punct
# 	- tokenize a file  ** make sure returns the correct number of sentences tokenized
# 	- tokenize a file of 10 collocs, batch size 3
# 	- tokenize a file of 10 collocs, start at 5
# 	- tokenize a file of 10 collocs, end at 5
