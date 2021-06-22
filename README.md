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

### Related links:
Colab notebook for training the models: https://colab.research.google.com/drive/1mG6x22ll75A1fjFQZUmGADPTp-j0XvXZ?usp=sharing \
Lemmatization, other final preprocessing steps: https://colab.research.google.com/drive/1BeRZOPHQfthUfK53uNcj4hWOLadActxu?usp=sharing


## Contents
- Configuration file `config.ini`
- Extraction of text from pdf files
    - `src/scripts/extraction/` 
- Preprocessing corpora of text extracted in HTML format via PDFBox
    - `src/html_preprocessor.py`
    - `src/tokenizer.py`
    - `src/preprocess.py`
- Training and loading word2vec and fastText models, and retrieving semantic 
neighbors for word associations
    - `src/word_embedder.py`
    - `src/train_word_embeddings.py`
- Validation through unittests
    - `tests/`

_note: additional instructions for using the treetagger tokenizer:_ \
In order to use the treetagger for tokenization, run the following from the src
dir:
```
wget https://www.cis.lmu.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.3.tar.gz
wget https://www.cis.lmu.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz
wget https://www.cis.lmu.de/~schmid/tools/TreeTagger/data/install-tagger.sh
wget http://corpus.leeds.ac.uk/mocky/russian.par.gz
sh install-tagger.sh
wget http://corpus.leeds.ac.uk/mocky/ru-table.tab
```

---
### Sources
[1]     A. Klimov, M. Kopotev and O. Kisselev, "Towards Intelligent Correction of Collocational Errors in Russian Novice Academic Texts in the CAT&kittens Writing Support Platform," 2020. \
[2]     M. Kopotev, D. Kormacheva and L. Pivovarova, "Constructional generalization over Russian collocations," Mémoires de la Société Néophilologique de Helsinki, pp. 121-140, 2016. \
[3]     S. Rodríguez-Fernández, L. Espinosa-Anke, R. Carlini and L. Wanner, "Semantics-Driven Recognition of Collocations Using Word Embeddings," in Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, Berlin, 2016.  \
[4]     E. Kochamar and T. Briscoe, "Detecting Learner Errors in the Choice of Content Worsd Using Compositional Distributional Semantics," in Proceedings of COLING 2014, the 25th International Meeting on Computational Linguistics, Dublin, 2014.  \
[5]     D. Alikaniotis and V. Raheja, "The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction," in 57th Annual Meeting of the Association for Computational Linguistics, Florence, 2019.  \
[6]     "PoA - Dist Sem," [Online]. Available: https://docs.google.com/document/d/1roYsQNsmZ2y2lLftn3QCgG293wpQBBlIH48KzK03GBk/edit?usp=sharing. 
