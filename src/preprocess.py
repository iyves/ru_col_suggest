import configparser
import logging
import os
import pickle

from joblib import delayed, Parallel
from pathlib import Path
from src.helpers import get_file_names, get_text
from src.preprocessor import HtmlPreprocessor

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
data_dir = config['PATHS']['data_dir']
log_dir = config['PATHS']['log_dir']
log_file = str(Path(log_dir, 'preprocessing.txt'))
logging.basicConfig(handlers=[logging.FileHandler(log_file, 'a', 'utf-8')],
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def preprocess_one(filename, file, target_txt_path, target_pickle_path) -> bool:
    """Preprocess one scholarly paper and save the text and tokens.

    :param filename: The name of the file to preprocessed.
    :param file: The path to the file to processed.
    :param target_txt_path: The path to save the preprocessed text.
    :param target_pickle_path: The path to save the preprocessed tokens.
    :return: True if successfully preprocessed the file, otherwise False
    """
    text = get_text(file)
    preprocessor = HtmlPreprocessor(text, filename)
    if not preprocessor.preprocess():
        logging.error("Error: Failed to preprocesses file: {}".format(filename))
        return False

    # Save preprocessed text as txt file
    output_file = Path(target_txt_path, filename + ".txt")
    if not os.path.isfile(output_file):
        with open(output_file, "w", encoding='utf8') as file:
            file.write(preprocessor.get_text())

    # Save tokens as pickle file
    output_file = Path(target_pickle_path, filename + ".pickle")
    if not os.path.isfile(output_file):
        with open(output_file, "wb") as file:
            tokens = list(preprocessor.tokenize(keep_pos=True, keep_punct=True))
            pickle.dump(tokens, file, pickle.HIGHEST_PROTOCOL)

    print(filename, end=", ")
    return True


# Preprocess all articles
for folder in 'Economics Education_and_psychology History Law Linguistics Sociology'.split():
    logging.info("Preprocessing papers from folder: {}".format(folder))
    print("\n\n", folder)

    # Set up the target directories, to where the preprocessed text will be saved
    source = str(Path(data_dir, "extracted/html/{}".format(folder)).resolve())
    target_txt = str(Path(data_dir, "preprocessed/text/{}".format(folder)).resolve())
    target_pickle = str(Path(data_dir, "preprocessed/pickle/{}".format(folder)).resolve())
    for target in [target_txt, target_pickle]:
        if not os.path.exists(target):
            os.makedirs(target)

    # Set up the files to preprocess
    files = [(os.path.splitext(os.path.basename(file))[0], file)
             for file in get_file_names(source, ".html")]

    # Preprocess each file in parallel
    element_information = Parallel(n_jobs=-1)(
        delayed(preprocess_one)(filename, file, target_txt, target_pickle) for filename, file in files)
    print("Done!")
