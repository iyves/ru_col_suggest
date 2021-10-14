from gensim.models import Word2Vec
import gensim
import os
import configparser

path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, 'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
models_dir = config['PATHS']['models_dir']
models = 'Economics Education_and_psychology History Law Linguistics Sociology supercybercat'
training_epochs = '30epx 100epx'

def main():
    model = Word2Vec.load("/scratch/project_2004882/cat/ru_col_suggest/src/models/w2v/Economics.model")
    for word in 'хельсинки_n встречать_v замок_n экономика_n доход_n'.split():
        try:
            sims = model.wv.most_similar(word, topn=10)
            print(word, '\n', sims)
        except KeyError:
            print(f'{word} not found in vocab')
    return True

if __name__ == '__main__':
    main()