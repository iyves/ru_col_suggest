import configparser
import logging
import os
import nltk
import re
from pathlib import Path


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


class PartialPreprocessor:
    '''
    Preprocess text files from old Liza's file format
    '''

    enumerated = r"[1-9]?[0-9]"
    whitespace = r" \n  "
    punctuation = r"!\"#$%&'()*+,-.\/:;<=>?@[\]^_`{|}~ʹ…〈〉«»—„“"
    roman_numeral = r"XVI"
    arabic_numeral = r"0-9"
    end_of_sentence = re.compile(r"([!?])|(:\s*$)")
    cyrillic = r"А-яЁё"
    has_lowercase = r"[а-яё]"
    paragraph_ends_with_period = r"\.\s*$"
    bold_and_italics = re.compile(r"(<b>.*?</b>)|(<i>.*?</i>)", re.MULTILINE | re.DOTALL)
    enumerated_example = re.compile(r"^(\({}\)|\s*—)".format(enumerated), re.MULTILINE)
    eol_hyphenation_in_paragraph = re.compile(r"[^\s]-\s*\n")
    sentence_broken_between_paragraphs = re.compile(r"({}.*{})"
        .format(has_lowercase, paragraph_ends_with_period))
    quoted_text = re.compile(r"[«“„'\"].*?[»”'\"“]", re.MULTILINE | re.DOTALL)
    footnote_reference = re.compile(r"\d+(?:[.,])|(?:[^\s\d])+?\d+(?:\s)", re.MULTILINE)
    not_alpha_cyrillic_or_punctuation = \
        re.compile(r'(?![{0}{1}])[{2}{3}{4}]*?'
                   r'[^{0}{1}{2}{3}{4}]+?'
                   r'.*?'
                   r'(?=$|[{0}{1}])'
                   .format(whitespace, punctuation,
                           roman_numeral, arabic_numeral,
                           cyrillic),
                   re.MULTILINE)
    arabic_numbers = re.compile(
        r"(?![{0}{1}])[{2}]+?(?=$|[{0}{1}])"
        .format(whitespace, punctuation, arabic_numeral),
        re.MULTILINE)
    roman_numbers = re.compile(r"(?![{0}{1}])[{2}]+?(?=$|[{0}{1}])"
        .format(whitespace, punctuation, roman_numeral),
        re.MULTILINE)
    
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
        self.text = text

    def get_text(self) -> str:
        return_string = []
        for line in self.text:
            return_string.append(' '.join(line))
        return ''.join(return_string)

    def remove_empty_lines_and_sentences(self) -> None:
        """Removes empty lines from text"""
        for i in reversed(range(len(self.text))):
            if len(self.text[i]) < 1:
                self.text.pop(i)
            else:
                for j in reversed(range(len(self.text[i]))):
                    if len(self.text[i][j]) < 1:
                        self.text[i].pop(j)
            
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
        # Hyphenation within lines
        self.text = [re.sub(self.eol_hyphenation_in_paragraph, "", line) for line in self.text]

        # for i in reversed(range(1, len(self.text))):
        #     if self.text[i - 1][-1] == "-":
        #         self.text[i - 1] = self.text[i - 1][:-1] + self.text[i]
        #         self.text.pop(i)

    def substitute_end_of_sentence_punctuation_with_period(self) -> None:
        """Replaces punctuation at the end of sentences with a period.
        Replaces all exclamation points and question marks with periods.
        Also, all commas followed by whitespace are replaces with periods.
        """
        self.text = [re.sub(self.end_of_sentence, ".", line) for line in self.text]

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
            elif not re.search(self.sentence_broken_between_paragraphs, self.text[i - 1]):
                self.text[i - 1] = self.text[i - 1] + ' ' + self.text[i]
                self.text.pop(i)

    def remove_quotations(self) -> None:
        """ Replaces all quoted strings of text with a QUOTE token.
        """
        self.text = [re.sub(self.quoted_text, "QUOTE", line) for line in self.text]


    def remove_styled_text(self) -> None:
        """ Replaces all bold and italicized text with a STYLE token.
        """
        self.text = [re.sub(self.bold_and_italics, "STYLE", line) for line in self.text]

    def remove_examples(self) -> None:
        """ Removes examples from the text.
        Removes any lines that begin with a parenthesized 1 or 2 digit number,
        and any lines that begin with whitespace followed by a long hyphen.
        """
        for i in reversed(range(len(self.text))):
            if re.search(self.enumerated_example, self.text[i]):
                self.text.pop(i)

    def remove_intext_references(self) -> None:
        """Remove in-text references.
        First, remove all sets of nested parenthesis and brackets -- these are
        typically how references are denoted.
        Second, remove all references to footnotes, i.e. numbers before a comma
        or period, and numbers at the end of words.
        """
        # Bracketed and parenthesized text
        self.text = [self.remove_nested_parenthesis_brackets(line) for line in self.text]

        # References to footnotes
        self.text = [re.sub(self.footnote_reference, "", line) for line in self.text]

    def remove_oov_tokens(self) -> None:
        """Replaces all strings of Out-Of-Vocabulary characters with an UNK
        token.
        OOV tokens are defined here as any words that contain a character that
        is not either a cyrillic character, a roman numeral, an arabic numeral,
        a whitespace character, or a punctuation mark.
        """
        self.text = [re.sub(self.not_alpha_cyrillic_or_punctuation,
                            "UNK", line) for line in self.text]

    def replace_numbers(self) -> None:
        """Replaces tokens consisting only of numbers with a NUM token.
        Tokens consisting only of 0-9 are replaced with a NUM_arab token.
        Tokens consisting only of XVI are replaced with a NUM_lat token.
        """
        self.text = [re.sub(self.arabic_numbers, "NUM_arab", line) for line in self.text]
        self.text = [re.sub(self.roman_numbers, "NUM_lat", line) for line in self.text]

    def break_lines_into_sentences(self) -> None:
        """Divide lines into sentences via the nltk sentence tokenizer.
        """
        self.text = [nltk.sent_tokenize(line, language="russian") for line in self.text]
        # print('-----------------\n', self.text)

    def remove_unwanted_sentences(self) -> int:
        """Removes all sentences with unwanted tokens.
        Sentences that contain UNK, QUOTE, or STYLE are considered unwanted.
        :return: The amount of sentences removed in total.
        """
        amount_removed = 0
        cyrillicc = re.compile(r"[{}]".format(self.cyrillic))
        for i in reversed(range(len(self.text))):
            for j in reversed(range(len(self.text[i]))):
                # Remove sentences with unwanted tokens
                sentence = self.text[i][j]
                if any(token in sentence for token in ["UNK", "QUOTE", "STYLE"]) \
                        or not re.search(cyrillicc, sentence):
                    self.text[i].pop(j)
                    amount_removed += 1

                # Remove empty lines
                if len(self.text[i]) < 1:
                    self.text.pop(i)
        return amount_removed

    def preprocess(self) -> bool:
        self.remove_eol_hyphenation()
        self.substitute_end_of_sentence_punctuation_with_period()
        self.join_broken_sentences()
        self.remove_quotations()
        if len(self.text) < 1:
            logging.error("Entire text was deleted after removing quotations: {}".format(self.filename))
            return False

        self.remove_styled_text()
        self.remove_examples()
        self.remove_intext_references()
        self.remove_oov_tokens()
        self.replace_numbers()

        # Phase 2 - List of Pages and Paragraphs and Sentences
        self.break_lines_into_sentences()
        self.remove_unwanted_sentences()
        # self.remove_empty_lines_and_sentences()
        if len(self.text) < 1:
            logging.error("Entire text was deleted after removing unwanted sentences: {}".format(self.filename))
            return False
        return True
