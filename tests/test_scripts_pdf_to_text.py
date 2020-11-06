import pytest

import src.scripts.pdf_to_text


class PdfParserTest:
    def test_single_column(self):
        # linguistics/0.pdf
        pass

    def test_double_columns(self):
        # linguistics/227.pdf
        pass

    def test_narrow_margins(self):
        # linguistics/420.pdf
        # linguistics/437.pdf
        pass

    def test_narrow_size(self):
        # linguistics/95.pdf
        pass

    def test_table_parses_body_text_only(self):
        # linguistics/04.pdf
        # linguistics/412.pdf
        # linguistics/15.pdf ?
        pass

    def test_indented_examples_parses_body_text_only(self):
        # linguistics/07.pdf
        pass

    def test_quote_example_parses(self):
        pass

    def test_italics_intext_examples(self):
        # linguistics/38.pdf
        # linguistics/447.pdf
        pass

    def test_old_church_slavonic(self):
        # Usually in bold font
        # linguistics/49.pdf
        # linguistics/368.pdf
        # linguistics/422.pdf
        pass

    def test_figure_pdf_parses_body_text_only(self):
        pass

    def test_footnote_references_parses_body_text_only(self):
        pass

