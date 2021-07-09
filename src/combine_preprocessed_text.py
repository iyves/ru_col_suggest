import codecs
import configparser
import nltk
import os
import re
import zipfile

from pathlib import Path
from src.helpers import get_file_names


# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
data_dir = config['PATHS']['data_dir']


"""Combine all preprocessed text from the CAT corpus into one file.
    Text should already by lemmatized and stored in .txt files with each lemmatized sentence
    delimited with a newline."""

files = []
for folder in 'Economics Education_and_psychology History Law Linguistics Sociology'.split():
    # Set up the target directories, to where the preprocessed text will be saved
    target_dir = str(Path(data_dir, "/preprocessed/text/{}".format(folder)).resolve())

    # Set up the files to preprocess
    files += [(os.path.splitext(os.path.basename(file))[0], file) for file in get_file_names(target_dir, ".txt")]

with codecs.open('CAT_sentences_full.txt', 'a', 'utf-8') as to_write:
    for file in files:
        for sentence in codecs.open(file[1], "r", "utf-8"):
            s = re.sub(' +', ' ', sentence)
            s = s.replace(" ", "").replace(" .", ".").replace(" ,", ",")
            lens = [len(word) for word in s.split() if word != "NUM_arab"]
            if len(lens) >= 5:
                avg = sum(lens) / len(lens)
                if avg > 5:
                    to_write.write(s)


"""Further preprocess text from the cybercat corpus, which have already been preprocessed
    via Liza's preprocessing script. Then, combine all preprocessed text into one file.
    
    Download the cybercat texts from gdrive, keeping them as .zip folders, partitioned by discipline (Linguistics, Economics, etc.). 
    This script assumes that the .zip files are stored in the directory `data/cybercat_texts`"""
cybercat_folder = str(Path(data_dir, 'cybercat_texts'))

# This preprocessing code comes from the HtmlPreprocessor class, and should be refactored later
phase1_end_of_sentence = re.compile(r"([!?])|(:\s*$)")
phase1_quoted_text = re.compile(r"[«“„'\"].*?[»”'\"“]", re.MULTILINE | re.DOTALL)
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

hanging_parenthesis = re.compile("\([^)]*?\(", re.MULTILINE)
hanging_brackets = re.compile("\[[^\]]*?\[]", re.MULTILINE)
footnote = re.compile("(?:[^ \n]+?)NUM_arab", re.MULTILINE)
start_with_capital = re.compile("^[А-ЯЁN]", re.MULTILINE)
start_with_period = re.compile("\.$", re.MULTILINE)


def substitute_end_of_sentence_punctuation_with_period(text):
    return re.sub(phase1_end_of_sentence, ".", text)


def remove_quotations(text):
    return re.sub(phase1_quoted_text, "QUOTE", text)


def close_hanging_parenthesis_brackets(text):
    text = re.sub(hanging_parenthesis, "", text)
    return re.sub(hanging_brackets, "", text)


def remove_nested_parenthesis_brackets(test_str):
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

def remove_oov_tokens(text):
    return re.sub(phase1_not_alpha_cyrillic_or_punctuation,
                  "UNK", text)


def replace_numbers(text):
    text = re.sub(phase1_arabic_numbers, "NUM_arab", text)
    return re.sub(phase1_roman_numbers, "NUM_lat", text)


# Phase 2 - List of Pages and Paragraphs and Sentences
def break_paragraphs_into_sentences(text):
    return nltk.sent_tokenize(text, language="russian")


def remove_unwanted_sentences(text):
    amount_removed = 0
    cyrillic = re.compile(r"[{}]".format(phase1_cyrillic))
    for i in reversed(range(len(text))):
        # Remove sentences with unwanted tokens
        sentence = text[i]
        if any(token in sentence for token in ["UNK", "QUOTE", "STYLE"]) \
                or not re.search(cyrillic, sentence):
            text.pop(i)
            amount_removed += 1
    return text
# end


with codecs.open('cybercat_sentences_full.txt', 'a', 'utf-8') as to_write:
    for zipped_folder in get_file_names(cybercat_folder, '.zip'):
        with zipfile.ZipFile(zipped_folder, 'r') as archive:
            for name in archive.namelist():
                with archive.open(name) as zipped_file:
                    text = ""
                    for line in codecs.iterdecode(zipped_file, 'utf8'):
                        line = line.strip()
                        if len(line) > 0:
                            text += ' ' + line
                    text = substitute_end_of_sentence_punctuation_with_period(text)
                    text = remove_quotations(text)
                    text = close_hanging_parenthesis_brackets(text)
                    text = remove_nested_parenthesis_brackets(text)
                    text = remove_oov_tokens(text)
                    text = replace_numbers(text)
                    text = break_paragraphs_into_sentences(text)
                    text = remove_unwanted_sentences(text)
                    for sentence in text:
                        s = re.sub(' +', ' ', sentence)
                        s = s.replace(" ", "").replace(" .", ".").replace(" ,", ",").replace("% ", "").replace("* ", "")
                        s = re.sub(footnote, "", s)
                        lens = [len(word) for word in s.split() if word != "NUM_arab"]
                        if len(lens) >= 5 and re.search(start_with_capital, s) and re.search(start_with_period, s) and '//' not in s:
                            avg = sum(lens) / len(lens)
                            if avg > 5:
                                to_write.write(s + "\n")
