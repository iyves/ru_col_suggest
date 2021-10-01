# Word embeddings for predicting scholarly word associations
The purpose of this project is to explore the viability of the usage of
word embeddings and transformers for erroneous collocation correction
in Russian academic texts.

Training data is sourced from the CAT database, which comprises of 3,600 Russian
 academic articles from six fields: Economics, Education and Psychology, Legal
 texts, Linguistics, History, and Sociology [1].

The code in this repository serves two purposes:
1. The implementation of the Russian collocation correction algorithm described 
in two of our previous research papers [1] [2].
2. The investigation of the usage of machine learning methods for the 
correction of Russian academic collocations.

Prior research into a similar task [3] [4], the automatic correction of grammatical errors, generally describe two to three steps: the extraction of collocations which form grammatical errors, the identification of the type of grammatical error, and the correction of the grammatical error. My research relates only to the last step: the correction of erroneous collocations. The goal of this research is to develop a writing platform to support learners of Russian as a foreign language and native Russian speakers who are developing the academic style of writing. A similar end-product, which is addressed towards the automatic correction of English grammatical errors is Grammarly [5].

The first part of this research is the extraction and preprocessing of text from a curated database of Russian academic papers, CAT and cybercat. Although the extraction and cleaning of text from the CAT database requires the usage of pdf extraction techniques, the text from cybercat was already extracted. The steps for preprocessing the extracted text can be reviewed in greater detail in the plan of action google document [6]. 

The preprocessed text is then used as training data for the second part of this research. Word embeddings and transformer-based language models are trained on GPUs via Google Colab. The word embedding models include: word2vec, Fasttext, GloVe, while the transformer-based language models are BERT, Elmo, and GPT-3. The methods used for the correction of academic collocation errors is the same for the word models, however, the language models use a fundamentally different approach (we are still experimenting with the best way to use language models for this task). 

## Project Contents
- [Setting up the config file](#setting-up-the-configini-file)
- [Text extraction from Russian academic papers in pdf format](#extracting-text-from-pdfs)
- [Text preprocessing](#preprocessing-of-text)
- [Training word embeddings and language models](#training-models)
- [Erroneous collocation correction with trained models](#using-trained-models)
- [Setting up a gcloud MySQL instance of cybercat](#setting-up-a-gcloud-mysql-instance-of-cybercat)
- [Model Evaluation](#model-evaluation)
- [Sources](#sources)

## Setting up the config.ini file
The config.ini file contains the full paths of the folders in which training data,
models, and logs reside. It also contains the connection information for attesting 
collocations against the cybercat database. The format of this file is as follows:
```text
[PATHS]
treetagger_dir=path/to/ru_col_suggest/src/treetaggers
models_dir=path/to/ru_col_suggest/src/models
log_dir=path/to/ru_col_suggest/data/log
data_dir=path/to/ru_col_suggest/data/

preprocessed_text_dir=path/to/ru_col_suggest/data/preprocessed/text

[SERVER]
DOMAIN=domain (i.e. cybercat)
HOST=server_ip
USER=username
PWD=pwd
```


## Extracting text from pdfs
In the `src/scripts/extraction` folder there are scripts for two methods of extracting text
from a pdf document:
- [Kutuzov's method](https://github.com/rusnlp/rusnlp/blob/033ef738e7791bb60afb398647c3d0512eaff4bc/code/web/backend/preprocessing/pdf_parser/make_txt_from_pdf.py): [pdf_to_text.py](src/scripts/extraction/pdf_to_text.py)
- [PDFBox method](https://pdfbox.apache.org/download.cgi): [parse_pdf_box.sh](src/scripts/extraction/parse_pdf_box.sh)

There is also a python script for checking the text extraction. It confirms whether or not there is
a matching extracted file for each pdf file.
- [check_parsed_files.py](src/scripts/extraction/check_parsed_files.py)

Prior to text extraction, pdfs of Russian academic papers must be stored in the 
`data/pdf` folder.

### Extraction methods
This project details two methods for extracting plain text from Russian academic
papers in pdf format. This task is deceivingly challenging due to the complex structure 
of pdfs [7]. The lack of standardized style formatting of academic papers further
complicates matters, as the presence and location of headers, footers, page numbers,
footnotes, and paper sections widely vary between disciplines and journals.

This project utilizes the PDFBox method, which allows for more fine-tuned parsing
by first transforming pdf documents into HTML. This keeps information about whitespace
and stylized text.

#### Kutuzov's method
This method involves the usage of [pdfminer3](https://pypi.org/project/pdfminer3/)
to extract text from pdf files.

The `pdf_to_text.py` script is a modified version of Kutuzov's pdf to text extraction script.
The modifications allow for multi-threaded extraction. The `-N` option controls the amount of 
threads to use in parallel.

The following command, when run from the `src/scripts/extraction` folder, will
extract text from pdfs in parallel on 4 threads:
```bash
python3 pdf_to_text.py --source-dir ../../data/pdf/ --target-dir ../../data/extracted/txt -N 4 -v
```

The script should automatically remove hyphenation from the text, but further preprocessing
may be necessary for removing headers, footers, page numbers, and in-text citations.

#### PDFBox method
The `pdfparse_pdf_box.sh` bash script involves the usage of PDFBox to covert pdf
documents into an html representation, which can be further processed. This method is used
for extracting the training data from CAT academic papers. This method can also extract only the text from
the pdf documents, but it will not automatically remove hyphenation.

Before running the script, PDFBox must be installed. This research uses the PDFBox standalone
command line tools: `pdfbox-app-2.0.21.jar`.
PDFBox can be installed from the [PDFBox homepage](https://pdfbox.apache.org/download.cgi).

The following command extracts the structure of all linguistics pdf documents into an html format on four threads in parallel.
Change the number at the end to modify the amount of threads to use.
```bash
bash ./parse_pdf_box.sh path/to/pdfbox-app-2.0.21.jar ../../../data/pdf/linguistics ../../../data/extracted/html/linguistics 4
```

### Validation of text extraction
The `check_parsed_files.py` script checks that there is a matching extracted file for each input pdf file.
Files extracted may be in .txt or .html format.

```bash
# Checking that pdfs are extracted into .txt files
python3 check_parsed_files.py --source-dir ../../../data/pdf --target-dir ../../../data/extracted/txt --file-type .txt -v

# Checking that pdfs are extracted into .html files
python3 check_parsed_files.py --source-dir ../../../data/pdf --target-dir ../../../data/extracted/html --file-type .html -v
```


## Preprocessing of text
There are three methods for preprocessing text. Liza's method was used for preprocessing text from
the cybercat database, whereas the PDFBox method was used for preprocessing text from the CAT database. 

The `src/scripts/preprocessing` folder includes two methods:
- [Kutuzov's method](https://github.com/akutuzov/webvectors/blob/db517610a5d9b5cb6c5f3fa3c55877c1291c0ec1/preprocessing/rus_preprocessing_udpipe.py): [kutuzov/rus_preprocessing_udpipe.py](src/scripts/preprocessing/kutuzov/rus_preprocessing_udpipe.py)
- Liza's method: [liza/CAT_preprocessing.py](src/scripts/preprocessing/liza/CAT_preprocessing.py)

The `src` folder includes files for preprocessing html files extracted via the PDFBox method.
The `preprocess.py` script uses the `HtmlPreprocessor` class to extract only the body text of the academic paper.
This occurs in three phases, totalling to sixteen steps:

&nbsp;&nbsp;&nbsp;&nbsp;__Phase 0: HTML - *HTML document in BeautifulSoup format*__
1. Break down the html document into pages and paragraphs, using [Kutuzov's unify_sym function](https://github.com/akutuzov/webvectors/blob/db517610a5d9b5cb6c5f3fa3c55877c1291c0ec1/preprocessing/rus_preprocessing_udpipe.py#L72)
for the unification of similar-looking symbols. 
     
   __Phase 1: List of Pages and Paragraphs - *Text[Page[Paragraph_index]]*__
2. Remove the references section.
3. Remove the keywords section and any preceding text (abstract & title).
4. Remove the header and footer from each page.
5. Remove footnotes, as well as the in-text references to footnotes.
6. Remove the End-of-Line hyphenation.
7. Replace punctuation at the end of sentences with a period.
8. Join sentences that were split at a paragraph or page boundary.
9. Replace all quoted strings of text with a `QUOTE` token.
10. Replace all bold and italicized text with a `STYLE` token.
11. Remove examples from the text. Examples are defined as any lines that begin with a parenthesized 1 or 2 digit number, and any lines that begin with whitespace followed by a long hyphen.
12. Remove in-text references.
13. Replace all strings of Out-Of-Vocabulary characters with an `UNK` token. OOV tokens are defined here as any words that contain a character that
        is not either a cyrillic character, a roman numeral, an arabic numeral,
        a whitespace character, or a punctuation mark.
14. Replace tokens consisting only of numbers with a `NUM` token.
        Tokens consisting only of 0-9 are replaced with a `NUM_arab` token.
        Tokens consisting only of XVI are replaced with a `NUM_lat` token.

    __Phase 2: List of Pages and Paragraphs and Sentences - *Text[Page[Paragraph[Sentence_index]]]*__
15. Break paragraphs into sentences with the [nltk sentence tokenizer](https://www.nltk.org/api/nltk.tokenize.html).
16. Remove all sentences with unwanted tokens. Sentences that contain a `UNK`, `QUOTE`, or `STYLE` token are considered unwanted.

The resulting text is stored in the .text field of the HtmlPreprocessor class as a list of pages, which is a list of paragraphs, which is a list of sentences. 
This text should be saved as in a .txt format with each sentence delimited with a newline for tokenization.

Next, the `tokenizer.py` script transforms the preprocessed sentences into tokens using one of
three methods:
- [TreeTagger method](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/): Default method
- [UDPipe / Kutuzov's method](https://github.com/akutuzov/webvectors/blob/db517610a5d9b5cb6c5f3fa3c55877c1291c0ec1/preprocessing/rus_preprocessing_udpipe.py)
- [MyStem method](https://pypi.org/project/pymystem3/): Uses Yandex's Mystem lemmatizer.

The default is to use the TreeTagger method, keep Part-of-Speech tags, and remove punctuation (only relevant for UDPipe).

The [`attest_collocation.ipynb`](attest_collocation.ipynb) colab file outlines the 
process of lemmatizing collocations.

Note: The UDPipe method will automatically download Kutuzov's latest trained UDPipe model. Must run `pip install ufal.udpipe`. [Tutorial](https://github.com/akutuzov/webvectors/blob/master/preprocessing/rusvectores_tutorial.ipynb) \
Note: The TreeTagger method requires additional installation, which can be done by running the following from the `src/treetagger`
folder (substitute "x" with the latest version https://www.cis.lmu.de/~schmid/tools/TreeTagger/):
```
wget https://www.cis.lmu.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-x.tar.gz
wget https://www.cis.lmu.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz
wget https://www.cis.lmu.de/~schmid/tools/TreeTagger/data/install-tagger.sh
wget http://corpus.leeds.ac.uk/mocky/russian.par.gz
sh install-tagger.sh
wget http://corpus.leeds.ac.uk/mocky/ru-table.tab
```

The `src/preprocess.py` script preprocesses all .html files in parallel. This script assumes that the
.html files, as extracted from pdfs via the PDFBox method, are partitioned by discipline (Linguistics, Economics, etc.), i.e. `data/extracted/html/Linguistics/`. 
The preprocessed text is saved to the directory `data/preprocessed/text`, and the lemmatized sentences are saved to
the directory `data/preprocessed/tokens`.

---

*Note: In the future, the following script should be refactored into the existing `preprocess.py` file.*

The [`src/combine_preprocessed_text.py`](src/combine_preprocessed_text.py) script combines all the preprocessed text into a 
single .txt file for ease of training. It is also used for further preprocessing of the 
cybercat texts, preprocessed via Liza's method. 

First, download the cybercat texts from gdrive, keeping them as .zip folders, partitioned by discipline (Linguistics, Economics, etc.). 
This script assumes that the .zip files are stored in the directory `data/cybercat_texts`.


This script does some additional preprocessing and removes corrupted
sentences. Corrupted sentences are those which which have an average word length less than
or equal to five. One again, this script should later be refactored to be included in the existing 
preprocessing script.

## Training models
After the preprocessed CAT and cybercat corpora are stored as individual text files in the format 
of lemmatized sentences delimited with a newline, the word embedding models may be trained.

This project learns four word embedding models:
- [Word2vec](https://radimrehurek.com/gensim/models/word2vec.html): Gensim's implementation of Google's CBOW algorithm.
- [FastText](https://radimrehurek.com/gensim/models/fasttext.html): Gensim's implementation of Facebook's FastText model.
- [GloVe](https://stackoverflow.com/questions/48962171/how-to-train-glove-algorithm-on-my-own-corpus): GloVe implementation in python via the [glove-python-binary](https://pypi.org/project/glove-python-binary/) library.
- [RoBERTa](https://github.com/huggingface/transformers): Huggingface implementation of the RuBERT model.

Training is done via google colab on High RAM GPUs. This process is detailed in the 
[`train_models.ipynb`](train_models.ipynb) colab file.

## Using trained models
Trained word embedding models can be used for erroneous collocation correction. There
are two different approaches, depending on the type of word embedding used.
The [`attest_collocation.ipynb`](attest_collocation.ipynb) colab file elucidates
these processes: lemmatization of input collocations, retrieval of collocation
replacement suggestions, and attesting of collocations.

### Static word embedding approach
This is the approach used for the w2v, fastText, and GloVe models. It follows our
algorithm outlined in [1] and [2]:
1. For a collocation consisting of n tokens, fix up to n-1 tokens.
2. Attest all possible collocation combinations, considering replacements for
un-fixed tokens that match the PoS tag of the original token.
3. Rank attested collocations by various collocatiability scores.
4. Select the collocations within a specified collocatiability threshold.

Attesting collocations requires connecting to the cybercat database.

### Dynamic word embedding approach
This approach applies to the RuBERT model, using masked language modeling to suggest
one or more corrections to a collocation, given the source sentence as context.

## Attesting Collocations and Calculating Collocatiability Scores
Collocations are attested on the cybercat database, which can be run locally or
through a cloud service. The `CollocationAttestor` class in the 
[`src/collocation_attestor.py`](src/collocation_attestor.py) file
contains the functionality  for verifying the existance of uni/bi/tri-grams
in the cybercat database.

This class is also responsible for the computation of 
collocatiability statistics, in particular: **PMI, t-score, and ngram frequency**.

Data returned from the `getCollocationReplacements` function can be passed
into the `writeResults` function to save the data as a .txt file in
csv format.

### Setting up a Gcloud MySQL instance of cybercat
This project used gcloud and **MySQL 8**. A backup of the local
cybercat database was stored on gdrive and transferred to gcloud. Then, the backup
was used to import the cybercat data onto a MySQL 8 server created in gcloud.

The first step is to create a bucket on gcloud (Storage>Cloud Storage) to store the cybercat backup. The 
backup file should already be in gdrive.

The colab document [`upload_cybercat_dump.ipynb`](upload_cybercat_dump.ipynb) shows how to copy a file from a gdrive
account to a gcloud account. This was used to transfer the MySQL backup file of the
cybercat database to gcloud for importing.

Create a MySQL database on gcloud using MySQL version 8.0 (Databases>SQL). Then, import the data from 
the bucket to the cloud database. After about 4.5 hours, the import should complete, and the bucket may be deleted.

To connect to the database, you must modify the server's network security configuration to
allow for external access (SQL>Primary Instance>Connections). This is detailed in the [gcloud documentation](https://cloud.google.com/sql/docs/mysql/authorize-networks).
The easiest method for testing is to add a network, and specify the IP address and subnet mask
for the computers from which the server will be accessed (Connections>Networking>Add Network). Adjust the `config.ini`
file accordingly.

## Model Evaluation
Collocations may be incorrect due to the collocations not existing in Russian, or 
to not being appropriate for academic style. That being said, the evaluation of erroneous collocation correction is difficult for two reasons:
1. There is no database to our knowledge for the evaluation of the closure task for Russian collocations.
2. There exists no database to our knowledge for the evaluation of Russian academic collocations.

To address these issues, we generate and manually label a dataset of erroneous collocations. These collocations are sourced
from a variety of methods:
- Collocation extraction from the kittens database
- Generation of collocations using the [CrossLexica](https://www.xl.gelbukh.com/) dictionary resource.

### Collocation extraction from kittens
The kittens database is created by our previous research [1][2]. It consists of texts
from students of Russian as a foreign language at various levels of proficiency.
Each line of the evaluation file contains a correct Russian academic collocation and 
for incorrect variants.

### Collocation extraction using the CrossLexica dictionary
The CrossLexica dictionary is a resource that provides synonyms, collocations, and other 
information for a given Russian input. Our method for generating erroneous collocation is as follows:
1. Extract the top 500 most frequent bigrams and trigrams that match a specified PoS filter:
    - Verb + Noun
    - Noun + Noun
    - Adjective + Noun
    - Verb + Verb
    - Verb + Infinitive
    - Verb + Preposition + Noun
    - Noun + Preposition + Noun
2. For each collocation, search CrossLexica for the synonyms of the first and last token in the collocation.
3. Fix either the first or last token and replace the other token with its synonyms.
4. Manually select the tokens that are not obviously incorrect.

The colab document [`generate_wrong_colloc.ipynb`](generate_wrong_colloc.ipynb) extrapolates on this process.

---
### Sources
[1]     A. Klimov, M. Kopotev and O. Kisselev, "Towards Intelligent Correction of Collocational Errors in Russian Novice Academic Texts in the CAT&kittens Writing Support Platform," 2020. \
[2]     M. Kopotev, D. Kormacheva and L. Pivovarova, "Constructional generalization over Russian collocations," Mémoires de la Société Néophilologique de Helsinki, pp. 121-140, 2016. \
[3]     S. Rodríguez-Fernández, L. Espinosa-Anke, R. Carlini and L. Wanner, "Semantics-Driven Recognition of Collocations Using Word Embeddings," in Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, Berlin, 2016.  \
[4]     E. Kochamar and T. Briscoe, "Detecting Learner Errors in the Choice of Content Worsd Using Compositional Distributional Semantics," in Proceedings of COLING 2014, the 25th International Meeting on Computational Linguistics, Dublin, 2014.  \
[5]     D. Alikaniotis and V. Raheja, "The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction," in 57th Annual Meeting of the Association for Computational Linguistics, Florence, 2019.  \
[6]     "PoA - Dist Sem," [Online]. Available: https://docs.google.com/document/d/1roYsQNsmZ2y2lLftn3QCgG293wpQBBlIH48KzK03GBk/edit?usp=sharing. \
[7]     A. Hashmi, F. Qayyum and M. Afzal, "Insights to the state-of-the-art PDF Extraction Techniques," in IPSI Transaction on Internet Research(16), pp. 1820-4503, 2020.
