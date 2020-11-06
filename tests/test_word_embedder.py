import logging
import pytest

from pathlib import Path
from src.helpers import get_text, get_file_names
from src.word_embedder import WordEmbedder


test_folder = Path("./fixtures/")


@pytest.fixture
def word_embedder(request) -> WordEmbedder:
    type = request.param
    return WordEmbedder(type)


class TestWordEmbedder:
    @pytest.mark.parametrize('word_embedder, model_type', [
        (WordEmbedder.Model.word2vec, WordEmbedder.Model.word2vec),
        (WordEmbedder.Model.fastText, WordEmbedder.Model.fastText)
    ], indirect=['word_embedder'])
    def test_Initialize_ShouldHaveDefaultValue(
            self, word_embedder, model_type, capsys):
        assert word_embedder.model_type == model_type
        assert word_embedder.model is None

        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""
