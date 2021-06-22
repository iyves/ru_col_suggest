import logging
import pytest

from pathlib import Path
from src.helpers import get_text, get_file_names
from src.html_preprocessor import HtmlPreprocessor


test_folder = Path("./fixtures/")


@pytest.fixture
def preprocessor(request) -> HtmlPreprocessor:
    filename = request.param
    return HtmlPreprocessor(text=get_text(test_folder/filename),
                            filename=filename)


class TestHtmlPreprocessor:
    def test_NoName_ShouldHaveDefaultFilename(self):
        preprocessor = HtmlPreprocessor(text="")
        assert preprocessor.filename == "missing_filename"

    @pytest.mark.parametrize('preprocessor, filename', [
        ('Linguistics_0.html', 'Linguistics_0.html'),
        ('Linguistics_505.html', 'Linguistics_505.html'),
        ('Linguistics_483.html', 'Linguistics_483.html')
        ], indirect=['preprocessor'])
    def test_ValidText_ShouldPass(
            self, preprocessor, filename, capsys):
        assert preprocessor.filename == filename
        assert preprocessor.phase == HtmlPreprocessor.Phase.HTML

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        ('Linguistics_0.html', 'Linguistics_0.html', 13),
        ('Linguistics_505.html', 'Linguistics_505.html', 11),
        ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_ValidTextExtractTextFromHtml_ShouldParseHtmlToListOfPagesAndParagraphs(
            self, preprocessor, filename, num_pages, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert "Page {}".format(num_pages) in captured.out
        assert "Page {}".format(num_pages+1) not in captured.out
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages, has_ref', [
        ('Linguistics_0.html', 'Linguistics_0.html', 12, True),
        ('Linguistics_505.html', 'Linguistics_505.html', 8, True),
        ('Linguistics_483.html', 'Linguistics_483.html', 8, False)
    ], indirect=['preprocessor'])
    def test_ValidTextRemoveReferences_ShouldRemoveReference(
            self, preprocessor, filename, num_pages, has_ref, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        assert preprocessor.remove_references() == has_ref
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert "Page {}".format(num_pages) in captured.out
        assert "Page {}".format(num_pages+1) not in captured.out
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages, has_keywords', [
        ('Linguistics_0.html', 'Linguistics_0.html', 12, True),
        ('Linguistics_505.html', 'Linguistics_505.html', 8, True),
        ('Linguistics_483.html', 'Linguistics_483.html', 8, True)
    ], indirect=['preprocessor'])
    def test_RemoveKeywords_ShouldRemoveKeywords(
            self, preprocessor, filename, num_pages, has_keywords, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        assert preprocessor.remove_keywords() == has_keywords
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert "Page {}".format(num_pages) in captured.out
        assert "Page {}".format(num_pages+1) not in captured.out
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages, num_headers_and_footers', [
        ('Linguistics_0.html', 'Linguistics_0.html', 12, 13),
        ('Linguistics_505.html', 'Linguistics_505.html', 8, 14),
        ('Linguistics_483.html', 'Linguistics_483.html', 8, 14)  # Not including first page's footer
    ], indirect=['preprocessor'])
    def test_RemoveHeadersAndFooters_ShouldRemoveHeadersAndFooters(
            self, preprocessor, filename, num_pages, num_headers_and_footers, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        assert preprocessor.remove_header_and_footer() == num_headers_and_footers
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages, num_footnotes', [
        ('Linguistics_0.html', 'Linguistics_0.html', 12, 1),
        ('Linguistics_505.html', 'Linguistics_505.html', 8, 0),
        ('Linguistics_483.html', 'Linguistics_483.html', 8, 1)
    ], indirect=['preprocessor'])
    def test_RemoveFootnotes_ShouldRemoveFootnotes(
            self, preprocessor, filename, num_pages, num_footnotes, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        assert preprocessor.remove_footnotes() == num_footnotes
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename', [
        ('Linguistics_0.html', 'Linguistics_0.html'),
        ('Linguistics_505.html', 'Linguistics_505.html'),
        ('Linguistics_483.html', 'Linguistics_483.html')
    ], indirect=['preprocessor'])
    def test_removeQuotations_ShouldNotPrintError(
            self, preprocessor, filename, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_quotations()
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages, num_empty_paragraphs', [
        ('Linguistics_0.html', 'Linguistics_0.html', 12, 0),
        ('Linguistics_505.html', 'Linguistics_505.html', 8, 0),
        ('Linguistics_483.html', 'Linguistics_483.html', 8, 0)
    ], indirect=['preprocessor'])
    def test_RemoveEmptyParagraphs_ShouldRemoveEmptyParagraphs(
            self, preprocessor, filename, num_pages, num_empty_paragraphs, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_quotations()
        assert preprocessor.remove_empty_paragraphs() == num_empty_paragraphs
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""


    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        ('Linguistics_0.html', 'Linguistics_0.html', 12),
        ('Linguistics_505.html', 'Linguistics_505.html', 8),
        ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_RemoveEolHyphenation_ShouldRemoveEolHyphenation(
            self, preprocessor, filename, num_pages, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_eol_hyphenation()
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        ('Linguistics_0.html', 'Linguistics_0.html', 12),
        ('Linguistics_505.html', 'Linguistics_505.html', 8),
        ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_SubstituteEndOfSentencePunctuationWithPeriod_ShouldNotHaveQuestionMarkOrExclamationPoint(
            self, preprocessor, filename, num_pages, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_eol_hyphenation()
        preprocessor.substitute_end_of_sentence_punctuation_with_period()
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert "!" not in captured.out
        assert "?" not in captured.out
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        ('Linguistics_0.html', 'Linguistics_0.html', 11),  # Last page gets joined with second to last page
        ('Linguistics_505.html', 'Linguistics_505.html', 8),
        ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_JoinBrokenSentences_ShouldNotPrintError(
        self, preprocessor, filename, num_pages, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_eol_hyphenation()
        preprocessor.substitute_end_of_sentence_punctuation_with_period()
        preprocessor.join_broken_sentences()
        preprocessor.remove_quotations()
        preprocessor.remove_empty_paragraphs()
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        ('Linguistics_0.html', 'Linguistics_0.html', 11),
        ('Linguistics_505.html', 'Linguistics_505.html', 8),
        ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_RemoveStyledText_ShouldNotPrintError(
        self, preprocessor, filename, num_pages, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_eol_hyphenation()
        preprocessor.substitute_end_of_sentence_punctuation_with_period()
        preprocessor.join_broken_sentences()
        preprocessor.remove_quotations()
        preprocessor.remove_empty_paragraphs()
        preprocessor.remove_styled_text()
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages, num_examples', [
        ('Linguistics_0.html', 'Linguistics_0.html', 11, 0),
        ('Linguistics_505.html', 'Linguistics_505.html', 8, 0),
        ('Linguistics_483.html', 'Linguistics_483.html', 8, 0)
    ], indirect=['preprocessor'])
    def test_RemoveExamples_ShouldRemoveCorrentNumberOfExamples(
            self, preprocessor, filename, num_pages, num_examples, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_eol_hyphenation()
        preprocessor.substitute_end_of_sentence_punctuation_with_period()
        preprocessor.join_broken_sentences()
        preprocessor.remove_quotations()
        preprocessor.remove_empty_paragraphs()
        preprocessor.remove_styled_text()
        assert preprocessor.remove_examples() == num_examples
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        ('Linguistics_0.html', 'Linguistics_0.html', 11),
        ('Linguistics_505.html', 'Linguistics_505.html', 8),
        ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_RemoveIntextReferences_ShouldNotRaiseErrors(
            self, preprocessor, filename, num_pages, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_eol_hyphenation()
        preprocessor.substitute_end_of_sentence_punctuation_with_period()
        preprocessor.join_broken_sentences()
        preprocessor.remove_quotations()
        preprocessor.remove_empty_paragraphs()
        preprocessor.remove_styled_text()
        preprocessor.remove_examples()
        preprocessor.remove_intext_references()
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        ('Linguistics_0.html', 'Linguistics_0.html', 11),
        ('Linguistics_505.html', 'Linguistics_505.html', 8),
        ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_RemoveOovTokens_ShouldNotRaiseErrors(
            self, preprocessor, filename, num_pages, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_eol_hyphenation()
        preprocessor.substitute_end_of_sentence_punctuation_with_period()
        preprocessor.join_broken_sentences()
        preprocessor.remove_quotations()
        preprocessor.remove_empty_paragraphs()
        preprocessor.remove_styled_text()
        preprocessor.remove_examples()
        preprocessor.remove_intext_references()
        preprocessor.remove_oov_tokens()
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        ('Linguistics_0.html', 'Linguistics_0.html', 11),
        ('Linguistics_505.html', 'Linguistics_505.html', 8),
        ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_ReplaceNumbers_ShouldNotRaiseErrors(
            self, preprocessor, filename, num_pages, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_eol_hyphenation()
        preprocessor.substitute_end_of_sentence_punctuation_with_period()
        preprocessor.join_broken_sentences()
        preprocessor.remove_quotations()
        preprocessor.remove_empty_paragraphs()
        preprocessor.remove_styled_text()
        preprocessor.remove_examples()
        preprocessor.remove_intext_references()
        preprocessor.remove_oov_tokens()
        preprocessor.replace_numbers()
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        ('Linguistics_0.html', 'Linguistics_0.html', 11),
        ('Linguistics_505.html', 'Linguistics_505.html', 8),
        ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_ReplaceNumbers_ShouldNotRaiseErrors(
            self, preprocessor, filename, num_pages, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_eol_hyphenation()
        preprocessor.substitute_end_of_sentence_punctuation_with_period()
        preprocessor.join_broken_sentences()
        preprocessor.remove_quotations()
        preprocessor.remove_empty_paragraphs()
        preprocessor.remove_styled_text()
        preprocessor.remove_examples()
        preprocessor.remove_intext_references()
        preprocessor.remove_oov_tokens()
        preprocessor.replace_numbers()
        preprocessor.break_paragraphs_into_sentences()
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS_AND_SENTENCES
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        ('Linguistics_0.html', 'Linguistics_0.html', 11),
        ('Linguistics_505.html', 'Linguistics_505.html', 8),
        ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_ReplaceNumbers_ShouldNotRaiseErrors(
            self, preprocessor, filename, num_pages, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_eol_hyphenation()
        preprocessor.substitute_end_of_sentence_punctuation_with_period()
        preprocessor.join_broken_sentences()
        preprocessor.remove_quotations()
        preprocessor.remove_empty_paragraphs()
        preprocessor.remove_styled_text()
        preprocessor.remove_examples()
        preprocessor.remove_intext_references()
        preprocessor.remove_oov_tokens()
        preprocessor.replace_numbers()
        preprocessor.break_paragraphs_into_sentences()
        assert preprocessor.remove_unwanted_sentences() != 0
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS_AND_SENTENCES
        assert len(preprocessor.text) == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        ('Linguistics_0.html', 'Linguistics_0.html', 11),
        ('Linguistics_505.html', 'Linguistics_505.html', 8),
        ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_Tokenize_ShouldNotRaiseErrors(
            self, preprocessor, filename, num_pages, capsys):
        assert preprocessor.filename == filename
        preprocessor.extract_text_from_html()
        preprocessor.remove_references()
        preprocessor.remove_keywords()
        preprocessor.remove_header_and_footer()
        preprocessor.remove_footnotes()
        preprocessor.remove_eol_hyphenation()
        preprocessor.substitute_end_of_sentence_punctuation_with_period()
        preprocessor.join_broken_sentences()
        preprocessor.remove_quotations()
        preprocessor.remove_empty_paragraphs()
        preprocessor.remove_styled_text()
        preprocessor.remove_examples()
        preprocessor.remove_intext_references()
        preprocessor.remove_oov_tokens()
        preprocessor.replace_numbers()
        preprocessor.break_paragraphs_into_sentences()
        tokenized_text = preprocessor.tokenize(keep_pos=True, keep_punct=True)
        tokenized_page_count = 0
        for page in tokenized_text:
            tokenized_page_count += 1

        assert preprocessor.remove_unwanted_sentences() != 0
        assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS_AND_SENTENCES
        assert len(preprocessor.text) == num_pages
        assert tokenized_page_count == num_pages

        preprocessor.print_text()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    @pytest.mark.parametrize('preprocessor, filename, has_references', [
        ('Linguistics_0.html', 'Linguistics_0.html', True),
        ('Linguistics_505.html', 'Linguistics_505.html', True),
        ('Linguistics_483.html', 'Linguistics_483.html', False)
    ], indirect=['preprocessor'])
    def test_Preprocess_ShouldNotRaiseErrors(
            self, preprocessor, filename, has_references, caplog):
        assert preprocessor.filename == filename
        with caplog.at_level(logging.INFO):
            assert preprocessor.preprocess()
        assert len(preprocessor.text) >= 1
        assert "Entire text was deleted" not in caplog.text
        print(caplog.text)
        assert ("Could not find the references" not in caplog.text) == has_references

    @pytest.mark.skip(reason="To be used for find-grained testing")
    @pytest.mark.parametrize('preprocessor, filename, num_pages', [
        # ('Linguistics_505.html', 'Linguistics_505.html', 11),
        # ('Sociology_505.html', 'Sociology_505.html', 8),
        ('Economics_14.html', 'Economics_14.html', 26)
        # ('Linguistics_483.html', 'Linguistics_483.html', 8)
    ], indirect=['preprocessor'])
    def test_detailed_preprocessing(
            self, preprocessor, filename, num_pages):
        assert preprocessor.filename == filename
        # preprocessor.extract_text_from_html()
        # preprocessor.remove_references()
        # preprocessor.remove_keywords()
        # preprocessor.remove_header_and_footer()
        # preprocessor.remove_footnotes()
        # preprocessor.remove_empty_paragraphs()
        # preprocessor.remove_eol_hyphenation()
        # preprocessor.substitute_end_of_sentence_punctuation_with_period()
        # preprocessor.join_broken_sentences()
        # preprocessor.remove_quotations()
        # preprocessor.remove_styled_text()
        # preprocessor.remove_examples()
        # preprocessor.remove_intext_references()
        # preprocessor.remove_oov_tokens()
        # preprocessor.replace_numbers()
        # preprocessor.break_paragraphs_into_sentences()
        # tokenized_text = preprocessor.tokenize(keep_pos=True, keep_punct=True)
        # tokenized_page_count = 0
        # for page in tokenized_text:
        #     tokenized_page_count += 1
        #
        # assert preprocessor.remove_unwanted_sentences() != 0
        # assert preprocessor.phase == HtmlPreprocessor.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS_AND_SENTENCES
        # assert len(preprocessor.text) == num_pages
        # assert tokenized_page_count == num_pages
        #
        assert preprocessor.preprocess()
        tokenized_text = preprocessor.tokenize(keep_pos=True, keep_punct=True)
        for text in tokenized_text:
            print(text)
        # preprocessor.print_text()
        # captured = capsys.readouterr()
        # assert captured.out != ""
        # assert captured.err == "
