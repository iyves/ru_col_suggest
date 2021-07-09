import codecs
import configparser
import os
import pandas as pd

from bs4 import BeautifulSoup, SoupStrainer
from pathlib import Path
from seleniumrequests import Firefox
from selenium import webdriver


# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
data_dir = config['PATHS']['data_dir']

options = webdriver.FirefoxOptions()
options.add_argument('start-maximized')
options.add_argument('--headless')
browser = Firefox(options=options, executable_path='C:/Program Files/geckodriver')


def get_synonyms(query: str):
    response = browser.request('POST', 'https://www.xl.gelbukh.com/', data={"query": query})
    full_html = response.text

    soup = BeautifulSoup(full_html, 'html.parser')

    # extract only lis
    lis = soup.find_all('li')
    synonyms = set()
    read = False
    for li in lis:
        # if there is an li with class noborder, see if it has an h3 with <font ...>Синонимы</font>
        li_a = li.find('a')
        if li_a and li_a.has_attr('name'):
            if li_a['name'] == "F_SYN":
                read = True
                continue
            else:
                read = False

        # Store subsequent lis if they don't begin with a whitespace
        if read and li.string and li.find('a'):
            synonym = li.find('a').string
            if synonym != query:
                synonyms.add(synonym)
    return list(synonyms)

synonyms = {}
bigrams = pd.read_csv(str(Path(data_dir, 'expanded_bigram_lemma_raw.csv')), encoding='utf8')

with codecs.open(str(Path(data_dir, 'expanded_wrong_bigrams.txt')), 'w+', encoding='utf-8') as out_file:
    out_file.write(",".join(["raw_frequency", "pos1", "pos2", "l1", "l2", "syn1", "syn2"]))
    for index, row in bigrams.iterrows():
        if row['l1'] not in synonyms:
            synonyms[row['l1']] = get_synonyms(row['l1'])
        if row['l2'] not in synonyms:
            synonyms[row['l2']] = get_synonyms(row['l2'])

        for synonym in synonyms[row['l1']]:
            out_file.write("\n" + ",".join([str(row['raw_frequency']), row["pos1"], row["pos2"],
                                            row["l1"], row["l2"], synonym, row["l2"]]))
        for synonym in synonyms[row['l2']]:
            out_file.write("\n" + ",".join([str(row['raw_frequency']), row["pos1"], row["pos2"],
                                            row["l1"], row["l2"], row['l1'], synonym]))

trigrams = pd.read_csv(str(Path(data_dir, 'expanded_trigram_lemma_raw.csv')), encoding='utf8')
with codecs.open(str(Path(data_dir, 'expanded_wrong_trigrams.txt')), 'w+', encoding='utf-8') as out_file:
    out_file.write(",".join(["raw_frequency", "pos1", "pos2", "pos3",
                             "l1", "l2", "l3", "syn1", "syn2", "syn3"]))
    for index, row in trigrams.iterrows():
        if row['l1'] not in synonyms:
            synonyms[row['l1']] = get_synonyms(row['l1'])
        if row['l3'] not in synonyms:
            synonyms[row['l3']] = get_synonyms(row['l3'])

        for synonym in synonyms[row['l1']]:
            out_file.write("\n" + ",".join([str(row['raw_frequency']), row["pos1"], row["pos2"], row["pos3"],
                                            row["l1"], row["l2"], row["l3"],
                                            synonym, row["l2"], row["l3"]]))
        for synonym in synonyms[row['l3']]:
            out_file.write("\n" + ",".join([str(row['raw_frequency']), row["pos1"], row["pos2"], row["pos3"],
                                            row["l1"], row["l2"], row["l3"],
                                            row["l1"], row["l2"], synonym]))
