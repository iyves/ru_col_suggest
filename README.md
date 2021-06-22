## Word embeddings for predicting scholarly word associations
This project contains:
- Configuration file `config.ini`
- Extraction of text from pdf files
    - `src/scripts/extraction/` 
- Preprocessing corpora of text extracted in HTML format via PDFBox
    - `src/preprocess.py`
    - `src/preprocessor.py`
- Training and loading word2vec and fastText models, and retrieving semantic 
neighbors for word associations
    - `src/word_embedder.py`
    - `src/train_word_embeddings.py`
- Validation through unittests
    - `tests/`
---
tokenizer instructions

!wget https://www.cis.lmu.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.3.tar.gz
!wget https://www.cis.lmu.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz
!wget https://www.cis.lmu.de/~schmid/tools/TreeTagger/data/install-tagger.sh
!wget http://corpus.leeds.ac.uk/mocky/russian.par.gz
!sh install-tagger.sh
!wget http://corpus.leeds.ac.uk/mocky/ru-table.tab