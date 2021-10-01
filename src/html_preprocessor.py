import configparser
import logging
import os
import nltk
import re

from bs4 import BeautifulSoup, SoupStrainer
from enum import Enum
from pathlib import Path
from typing import Iterable

import scripts.preprocessing.kutuzov.rus_preprocessing_udpipe as kutuzov


# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'preprocessing.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class HtmlPreprocessor:
    """Preprocesses Russian text in HTML format extracted from a scholarly paper
    via the PDFBox tool.
    """

    class Phase(Enum):
        """The current representation of the scholarly text.
        HTML - HTML document in BeautifulSoup format
        LISTS_OF_PAGES_AND_PARAGRAPHS - Text[Page[Paragraph_index]]
        LISTS_OF_PAGES_AND_PARAGRAPHS_AND_SENTENCES - Text[Page[Paragraph[Sentence_index]]]
        """
        HTML = 0
        LISTS_OF_PAGES_AND_PARAGRAPHS = 1
        LISTS_OF_PAGES_AND_PARAGRAPHS_AND_SENTENCES = 2

    # Regex patterns
    phase0_new_paragraph = re.compile(r"\n+\s+")
    phase0_paragraph = re.compile(r"<p>.*?</p>", re.MULTILINE | re.DOTALL)

    phase1_references_start = r"((([1-9][0-9]*(\.[1-9][0-9]*)*)|[ilvVxX]+)(\.?)(\s+))?"
    phase1_references_en = r"(<[ib]>)*[/s/n]*([rR]eference|REFERENCE|[bB]ibliography|BIBLIOGRAPHY)"
    phase1_references_ru = r"(<[ib]>)*[/s/n]*(Л ?и ?т ?е ?р ?а ?т ?у ?р ?а|Л ?И ?Т ?Е ?Р ?А ?Т ?У ?Р ?А|" \
                           r"С ?п ?и ?с ?о ?к +л ?и ?т ?е ?р ?а ?т ?у ?р ?ы|С ?П ?И ?С ?О ?К +Л ?И ?Т ?Е ?Р ?А ?Т ?У ?Р ?Ы|" \
                           r"С ?п ?и ?с ?о ?к +и ?с ?т ?о ?ч ?н ?и ?к ?о ?в|С ?П ?И ?С ?О ?К +И ?С ?Т ?О ?Ч ?Н ?И ?К ?О ?В|" \
                           r"И ?с ?т ?о ?ч ?н ?и ?к ?и|И ?С ?Т ?О ?Ч ?Н ?И ?К ?И|" \
                           r"Б ?и ?б ?л ?и ?о ?г ?р ?а ?ф ?и|Б ?И ?Б ?Л ?И ?О ?Г ?Р ?А ?Ф ?И)"
    phase1_references_english = re.compile(r"^{}{}.{{0,80}}$".format(
        phase1_references_start, phase1_references_en), re.MULTILINE | re.DOTALL)
    phase1_references_russian = re.compile(r"^{}{}.{{0,80}}$".format(
        phase1_references_start, phase1_references_ru), re.MULTILINE | re.DOTALL)
    phase1_keyword_section = re.compile(r"(<[bi]>)*[/s/n]*(keywords|ключевые ?слова)",
                                     re.IGNORECASE)
    phase1_enumerated = r"[1-9]?[0-9]"
    phase1_footnote = re.compile(r"^{} .+".format(phase1_enumerated))
    phase1_quoted_text = re.compile(r"[«“„'\"].*?[»”'\"“]", re.MULTILINE | re.DOTALL)
    phase1_eol_hyphenation_in_paragraph = re.compile(r"[^\s]-\s*\n")
    phase1_end_of_sentence = re.compile(r"([!?])|(:\s*$)")
    phase1_paragraph_ends_with_period = r"\.\s*$"
    phase1_has_lowercase = r"[а-яё]"
    phase1_sentence_broken_between_paragraphs = re.compile(r"({}.*{})"
        .format(phase1_has_lowercase, phase1_paragraph_ends_with_period))
    phase1_bold_and_italics = re.compile(r"(<b>.*?</b>)|(<i>.*?</i>)", re.MULTILINE | re.DOTALL)
    phase1_enumerated_example = re.compile(r"^(\({}\)|\s*—)"
                                    .format(phase1_enumerated), re.MULTILINE)
    phase1_footnote_reference = re.compile(r"\d+(?:[.,])|(?:[^\s\d])+?\d+(?:\s)", re.MULTILINE)

    phase1_whitespace = r" \n  "
    phase1_punctuation = r"!\"#$%&'()*+,-.\/:;<=>?@[\]^_`{|}~ʹ…〈〉«»—„“"
    phase1_roman_numeral = r"XVI"
    phase1_arabic_numeral = r"0-9"
    phase1_cyrillic = r"А-яЁё"
    # Selects only words that are non-alphacyrillic, numerals, or punctuation
    # i.e. non-Russian words, corrupted text, etc.
    phase1_not_alpha_cyrillic_or_punctuation = \
        re.compile(r'(?![{0}{1}])[{2}{3}{4}]*?'
                   r'[^{0}{1}{2}{3}{4}]+?'
                   r'.*?'
                   r'(?=$|[{0}{1}])'
                   .format(phase1_whitespace, phase1_punctuation,
                           phase1_roman_numeral, phase1_arabic_numeral,
                           phase1_cyrillic),
                   re.MULTILINE)

    phase1_arabic_numbers = re.compile(
        r"(?![{0}{1}])[{2}]+?(?=$|[{0}{1}])"
            .format(phase1_whitespace, phase1_punctuation, phase1_arabic_numeral),
        re.MULTILINE)
    phase1_roman_numbers = re.compile(
        r"(?![{0}{1}])[{2}]+?(?=$|[{0}{1}])"
            .format(phase1_whitespace, phase1_punctuation, phase1_roman_numeral),
        re.MULTILINE)

    # From https://stackoverflow.com/questions/14596884/remove-text-between-and-in-python/14598135#14598135
    @classmethod
    def remove_nested_parenthesis_brackets(cls, test_str):
        ret = ''
        skip1c = 0
        skip2c = 0
        for i in test_str:
            if i == '[':
                skip1c += 1
            elif i == '(':
                skip2c += 1
            elif i == ']' and skip1c > 0:
                skip1c -= 1
            elif i == ')' and skip2c > 0:
                skip2c -= 1
            elif skip1c == 0 and skip2c == 0:
                ret += i
        return ret

    def __init__(self, text: str, filename: str = "missing_filename"):
        """Stores the filename and <body> subtree.

        :param text: The extracted HTML document as a string after passing a
            Russian scholarly paper in PDF format through the PDFBox tool.
        :param filename: The name of the file associated with the extracted
            HTML document.
        """
        self.filename = filename
        self.text = BeautifulSoup(text, 'html.parser',
                                  parse_only=SoupStrainer("body"))
        self.phase = self.Phase.HTML

    def get_text(self) -> str:
        """Return the preprocessed text as a string.

        :return: The text in a string format.
        """
        return_string = []
        if self.phase == self.Phase.HTML:
            return_string.append(str(self.text))
        else:
            for idx, page in enumerate(self.text, 1):
                for paragraph in page:
                    if self.phase == self.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS:
                        return_string.append(str(paragraph))
                    elif self.phase == self.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS_AND_SENTENCES:
                        return_string.append("\n".join(paragraph))
                    else:
                        return_string.append(" ".join(str(paragraph)))
                    return_string.append("\n")
        return "".join(return_string)

    def print_text(self) -> None:
        """Prints the contents of the text that is being preprocessed to stdout,
            considering the preprocessing step that the text has completed.
        """
        logging.info("Text for {}:".format(self.filename))
        if self.phase == self.Phase.HTML:
            print(str(self.text))
        else:
            for idx, page in enumerate(self.text, 1):
                print("\n", "-" * 50, "Start of Page", idx, "-" * 50, "\n")
                for paragraph in page:
                    if self.phase == self.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS:
                        print(str(paragraph))
                    elif self.phase == self.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS_AND_SENTENCES:
                        for sentence in paragraph:
                            print(sentence)
                    else:
                        print(" ".join(str(paragraph)))
                    print("\n-----")
                print("\n", "-" * 50, "End of Page", idx, "-" * 50, "\n")

    # Preprocessing functions for phase 0 (HTML document)
    def __extract_paragraphs_from_html(self, paragraphs: Iterable[str]) -> Iterable[str]:
        """Helper function for extract_text_from_html. Fully breaks down the
            paragraph tags in a page into a list of paragraphs.

        This function addresses instances where BeautifulSoup fails to identify
        all of the paragraphs in a page, i.e. when two blocks of texts are
        separated by a new line rather than <p> tags.

        :param paragraphs: A list of extracted paragraphs (i.e. <p> subtrees)
            from a page.
        :return: A list of all paragraphs from the page without the <p> tags.
        """
        paragraphs = [re.sub(self.phase0_new_paragraph, "</p><p>", para) for para in paragraphs]
        paragraphs = re.findall(self.phase0_paragraph, "".join(paragraphs))
        paragraphs = [para[3:-4].strip() for para in paragraphs]  # Remove the wrapping <p></p>
        return paragraphs

    def extract_text_from_html(self) -> None:
        """Extracts the text from each page of the HTML subtree.

        An HTML document is broken down into pages, identified as each <div> in
        the <body> subtree. Each page is further roughly broken down into lists
        of paragraphs, which are all <p> subtrees in the <div> of a page.

        For each list of paragraphs (page), the text is fully broken down into
        paragraphs and the text is extracted. Each paragraph is passed through
        Kutuzov's preprocessing function for the standardization of symbols.

        The result is a representation of the text as a list of pages, with each
        page being a list of paragraphs, aka Phase 1.
        """
        self.text = [self.__extract_paragraphs_from_html([kutuzov.unify_sym(str(paragraph))
                                                          for paragraph in page.div.find_all('p', recursive=False)])
                             for page in self.text.body.find_all('div', recursive=False)]
        self.phase = self.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS

    # Preprocessing functions for Phase 1(List of pages and paragraphs)
    def remove_empty_paragraphs(self) -> int:
        """Removes all empty paragraphs and empty pages from a Phase 1 text.

        Starts from the last page to the first page, and the last paragraph of a
        page to the first paragraph of a page to avoid skipping paragraphs.

        First, remove the page if it is empty. If not, remove all empty
        paragraphs. If the resulting page is now empty, remove it as well.

        :return: The amount of paragraphs and empty pages removed in total.
        """
        amount_removed = 0
        if self.phase != self.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS:
            logging.error("Can't remove empty paragraphs from a text in phase {}"
                          .format(self.phase))
            return 0

        for i in reversed(range(len(self.text))):
            if len(self.text[i]) < 1:
                self.text.pop(i)
                amount_removed += 1
            else:
                for j in reversed(range(len(self.text[i]))):
                    if self.text[i][j].strip() == "":
                        self.text[i].pop(j)
                        amount_removed += 1
                if len(self.text[i]) < 1:
                    self.text.pop(i)
        return amount_removed

    def remove_references(self) -> bool:
        """Removes the references section from text parsed from a Russian
        scholarly paper.

        :return: True if references section found and removed, otherwise False
        """
        # Try to find references in Russian first
        for i in range(len(self.text)):
            for j in range(len(self.text[i])):
                if re.search(self.phase1_references_russian, self.text[i][j]):
                    self.text[i] = self.text[i][:j]
                    self.text = self.text[:i + 1]
                    return True

        # Then try to find the references in English
        for i in range(len(self.text)):
            for j in range(len(self.text[i])):
                if re.search(self.phase1_references_english, self.text[i][j]):
                    self.text[i] = self.text[i][:j]
                    self.text = self.text[:i + 1]
                    return True
        return False

    def remove_keywords(self) -> bool:
        """Removes the keywords section and any preceding text.

        Searches the first two pages for keywords, starting from the last
        paragraph and working towards the first paragraph.

        :return: True if keywords were removed, otherwise false.
        """
        # Check that there is at least one page
        if len(self.text) < 1:
            logging.error("No pages from which to remove keywords for file: {}"
                          .format(self.filename))
            return False

        # Check page 1 for keywords
        for j in reversed(range(len(self.text[0]))):
            if re.search(self.phase1_keyword_section, self.text[0][j]):
                self.text[0] = self.text[0][j + 1:]
                return True

        # Check page 2 for keywords
        if len(self.text) < 2:
            return False

        for j in reversed(range(len(self.text[1]))):
            if re.search(self.phase1_keyword_section, self.text[1][j]):
                self.text[1] = self.text[1][j + 1:]
                self.text.pop(0)
                return True
        return False

    def remove_header_and_footer(self) -> int:
        """Simplistic method for removing the header and footer from each page.

        Removes the first (and possibly second) line of each page if the
        lines do not have the newline character (i.e. '\n'). Note that this
        method is simplistic, and will fail in some cases.

        :return: The amount of headers and footers removed in total.
        """
        amount_removed = 0
        for i in reversed(range(len(self.text))):
            # Check that the page has at least one paragraph first
            if len(self.text[i]) < 1:
                self.text.pop(i)
            else:
                if "\n" not in re.sub("\n</i>", "", self.text[i][0]):
                    if len(self.text[i]) > 1:
                        if "\n" not in re.sub("\n</i>", "", self.text[i][1]):
                            self.text[i] = self.text[i][2:]  # Header and footer
                            amount_removed += 2
                        else:
                            self.text[i] = self.text[i][1:]  # Either just header or footer
                            amount_removed += 1
                    else:
                        self.text.pop(i)
        return amount_removed

    def remove_footnotes(self) -> int:
        """Removes all footnotes from each page.

        Searches each page for the first footnote and then removes all of the
        remaining lines in the page.

        :return: The amount of pages from which footnotes were removed.
        """
        amount_removed = 0
        for i in range(len(self.text)):
            for j in range(len(self.text[i])):
                if re.search(self.phase1_footnote, self.text[i][j]):
                    self.text[i] = self.text[i][:j]
                    amount_removed += 1
                    break
        return amount_removed

    def remove_quotations(self) -> None:
        """ Replaces all quoted strings of text with a QUOTE token.
        """
        for page in self.text:
            for j in range(len(page)):
                page[j] = re.sub(self.phase1_quoted_text, "QUOTE", page[j])

    def remove_eol_hyphenation(self) -> None:
        """Removes end-of-line hyphenation.

        First, join words that were split in half at the right of the page.

        Next, join words that were split in half at the end of a paragraph, with
        the second half of the word occurring in the next paragraph. This does
        not include words split at a page boundary.

        Finally, join words that were split in half at the end a page, with the
        second half of the word occurring in the first paragraph of the next
        page.
        """
        # Hyphenation within paragraphs
        for page in self.text:
            for j in range(len(page)):
                page[j] = re.sub(self.phase1_eol_hyphenation_in_paragraph, "", page[j])
                page[j] = re.sub("\n", " ", page[j])

        # Hyphenation between paragraphs
        for page in self.text:
            for j in reversed(range(1, len(page))):
                if page[j - 1][-1] == "-":
                    page[j - 1] = page[j - 1][:-1] + page[j]
                    page.pop(j)

        # Hyphenation between pages:
        for i in reversed(range(1, len(self.text))):
            if self.text[i - 1][-1][-1] == "-":
                self.text[i - 1][-1] = self.text[i - 1][-1][:-1] + self.text[i][0]

                if len(self.text[i]) > 1:
                    self.text[i] = self.text[i][1:]
                else:
                    self.text.pop(i)

    def substitute_end_of_sentence_punctuation_with_period(self) -> None:
        """Replaces punctuation at the end of sentences with a period.

        Replaces all exclamation points and question marks with periods.
        Also, all commas followed by whitespace are replaces with periods.
        """
        for i in range(len(self.text)):
            self.text[i] = [re.sub(self.phase1_end_of_sentence, ".", para)
                            for para in self.text[i]]

    def join_broken_sentences(self) -> None:
        """Joins sentences that were split at a paragraph or page boundary.

        First, search each page from the last paragraph to the second paragraph,
        so as to not skip paragraphs. If a paragraph begins with a sentence that
        starts with a lower-case character, then join the paragraph with the
        previous paragraph.

        Second, search for broken sentences between pages. Join the broken
        sentence to the first page.
        """
        # Sentence broken between paragraphs
        for i in reversed(range(len(self.text))):
            # Page with no paragraphs
            if len(self.text[i]) < 1:
                self.text.pop(i)
            else:
                for j in reversed(range(1, len(self.text[i]))):
                    # Empty paragraph
                    if len(self.text[i][j].strip()) < 1:
                        self.text[i].pop(j)
                    elif not re.search(self.phase1_sentence_broken_between_paragraphs, self.text[i][j - 1]):
                        self.text[i][j - 1] = self.text[i][j - 1] + ' ' + self.text[i][j]
                        self.text[i].pop(j)

        # Sentence broken between pages:
        for i in reversed(range(1, len(self.text))):
            if not re.search(self.phase1_paragraph_ends_with_period, self.text[i - 1][-1]):
                self.text[i - 1][-1] = self.text[i - 1][-1] + " " + self.text[i][0]

                if len(self.text[i]) > 1:
                    self.text[i] = self.text[i][1:]
                else:
                    self.text.pop(i)

    def remove_styled_text(self) -> None:
        """ Replaces all bold and italicized text with a STYLE token.
        """
        for page in self.text:
            for j in range(len(page)):
                page[j] = re.sub(self.phase1_bold_and_italics, "STYLE", page[j])

    def remove_examples(self) -> int:
        """ Removes examples from the text.

        Removes any lines that begin with a parenthesized 1 or 2 digit number,
        and any lines that begin with whitespace followed by a long hyphen.

        :return: The amount of examples removed in total.
        """
        amount_removed = 0
        # Lines that start with enumerated
        for i in reversed(range(len(self.text))):
            for j in reversed(range(len(self.text[i]))):
                if re.search(self.phase1_enumerated_example, self.text[i][j]):
                    self.text[i].pop(j)
                    amount_removed += 1
            if len(self.text[i]) < 1:
                self.text.pop(i)
        return amount_removed

    def remove_intext_references(self) -> None:
        """Remove in-text references.

        First, remove all sets of nested parenthesis and brackets -- these are
        typically how references are denoted.

        Second, remove all references to footnotes, i.e. numbers before a comma
        or period, and numbers at the end of words.
        """
        # Bracketed and parenthesized text
        for i in range(len(self.text)):
            self.text[i] = [self.remove_nested_parenthesis_brackets(para)
                            for para in self.text[i]]

        # References to footnotes
        for i in range(len(self.text)):
            self.text[i] = [re.sub(self.phase1_footnote_reference, "", para)
                            for para in self.text[i]]

    def remove_oov_tokens(self) -> None:
        """Replaces all strings of Out-Of-Vocabulary characters with an UNK
        token.

        OOV tokens are defined here as any words that contain a character that
        is not either a cyrillic character, a roman numeral, an arabic numeral,
        a whitespace character, or a punctuation mark.
        """
        for i in range(len(self.text)):
            self.text[i] = [re.sub(self.phase1_not_alpha_cyrillic_or_punctuation,
                                   "UNK", para) for para in self.text[i]]

    def replace_numbers(self) -> None:
        """Replaces tokens consisting only of numbers with a NUM token.

        Tokens consisting only of 0-9 are replaced with a NUM_arab token.
        Tokens consisting only of XVI are replaced with a NUM_lat token.
        """
        for i in range(len(self.text)):
            self.text[i] = [re.sub(self.phase1_arabic_numbers, "NUM_arab", para) for para in self.text[i]]
            self.text[i] = [re.sub(self.phase1_roman_numbers, "NUM_lat", para) for para in self.text[i]]

    def break_paragraphs_into_sentences(self) -> None:
        """Divide paragraphs into sentences via the nltk sentence tokenizer.
        """
        for i in range(len(self.text)):
            self.text[i] = [nltk.sent_tokenize(paragraph, language="russian")
                            for paragraph in self.text[i]]

        self.phase = self.Phase.LISTS_OF_PAGES_AND_PARAGRAPHS_AND_SENTENCES

    # Preprocessing functions for Phase 2 (List of pages, paragraphs, and sentences)
    def remove_unwanted_sentences(self) -> int:
        """Removes all sentences with unwanted tokens.

        Sentences that contain UNK, QUOTE, or STYLE are considered unwanted.

        :return: The amount of sentences removed in total.
        """
        amount_removed = 0
        cyrillic = re.compile(r"[{}]".format(self.phase1_cyrillic))
        for i in reversed(range(len(self.text))):
            for j in reversed(range(len(self.text[i]))):
                for k in reversed(range(len(self.text[i][j]))):
                    # Remove sentences with unwanted tokens
                    sentence = self.text[i][j][k]
                    if any(token in sentence for token in ["UNK", "QUOTE", "STYLE"]) \
                            or not re.search(cyrillic, sentence):
                        self.text[i][j].pop(k)
                        amount_removed += 1

                # Remove empty paragraphs
                if len(self.text[i][j]) < 1:
                    self.text[i].pop(j)
            # Remove empty pages
            if len(self.text[i]) < 1:
                self.text.pop(i)
        return amount_removed

    def preprocess(self) -> bool:
        # Phase 0 - HTML
        self.extract_text_from_html()

        # Phase 1 - List of Pages and Paragraphs
        if not self.remove_references():
            logging.info("Could not find the references section for file: {}".format(self.filename))
        if not self.remove_keywords():
            logging.info("Could not find the keywords section for file: {}".format(self.filename))
        self.remove_empty_paragraphs()
        if len(self.text) < 1:
            logging.error("Entire text was deleted after removing keywords: {}".format(self.filename))
            return False

        self.remove_header_and_footer()
        self.remove_footnotes()
        self.remove_empty_paragraphs()
        if len(self.text) < 1:
            logging.error("Entire text was deleted after removing footnotes: {}".format(self.filename))
            return False

        self.remove_eol_hyphenation()
        self.substitute_end_of_sentence_punctuation_with_period()
        self.join_broken_sentences()
        self.remove_quotations()
        self.remove_empty_paragraphs()
        if len(self.text) < 1:
            logging.error("Entire text was deleted after removing quotations: {}".format(self.filename))
            return False

        self.remove_styled_text()
        self.remove_examples()
        self.remove_intext_references()
        self.remove_oov_tokens()
        self.replace_numbers()

        # Phase 2 - List of Pages and Paragraphs and Sentences
        self.break_paragraphs_into_sentences()
        self.remove_unwanted_sentences()
        if len(self.text) < 1:
            logging.error("Entire text was deleted after removing unwanted sentences: {}".format(self.filename))
            return False
        return True

