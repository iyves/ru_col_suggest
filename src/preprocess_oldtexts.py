import configparser
import logging
import os

from joblib import delayed, Parallel
from pathlib import Path
from helpers import get_file_names, get_text
from partial_preprocessor import PartialPreprocessor

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
data_dir = config['PATHS']['data_dir']
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'preprocess.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def preprocess_one(filename, file, target_txt_path) -> bool:
    """Preprocess one scholarly paper and save the text and tokens.
    :param filename: The name of the file to preprocessed.
    :param file: The path to the file to processed.
    :param target_txt_path: The path to save the preprocessed text.
    :param target_token_path: The path to save the preprocessed tokens.
    :param method: The method for lemmatizing the text. Default: TREETAGGER
    :return: True if successfully preprocessed the file, otherwise False
    """
    text = get_text(file)
    preprocessor = PartialPreprocessor(text, filename)
    if not preprocessor.preprocess():
        logging.error("Error: Failed to preprocesses file: {}".format(filename))
        return False

    # Save preprocessed text as txt file
    output_txt_file = Path(target_txt_path, filename + ".txt")
    if not os.path.isfile(output_txt_file):
        with open(output_txt_file, "w", encoding='utf8') as file:
            file.write(preprocessor.get_text())

    logging.info("Preprocessed file: {}".format(filename))
    print(filename, end=", ")
    return True


# Preprocess all articles
for folder in 'law_cleaned '.split():
    logging.info("Preprocessing papers from folder: {}".format(folder))
    print("\n\n", folder)

    # Set up the target directories, to where the preprocessed text will be saved
    source = str(Path(data_dir, "preprocessed/old_texts/{}".format(folder)).resolve())
    target_txt = str(Path(data_dir, "preprocessed/old_texts_cleaned/{}".format(folder)).resolve())
    for target in [target_txt, ]:
        if not os.path.exists(target):
            os.makedirs(target)

    # Set up the files to preprocess
    files = [(os.path.splitext(os.path.basename(file))[0], file)
             for file in get_file_names(source, ".txt")]

    # Preprocess each file in parallel
    # for filename, file in files:
    #     preprocess_one(filename, file, target_txt)
    element_information = Parallel(n_jobs=-1)(
        delayed(preprocess_one)(filename, file, target_txt) for filename, file in files)
    print("Done!")