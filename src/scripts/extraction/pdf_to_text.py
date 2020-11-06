"""Extracts the text from a .pdf document.
This file is adapted from research by:
@article{bakarovrussian,
  title={Russian Computational Linguistics: Topical Structure in 2007-2017 Conference Papers},
  journal={Komp'yuternaya Lingvistika i Intellektual'nye Tekhnologii},
  year={2018},
  author={Bakarov, Amir and Kutuzov, Andrey and Nikishina, Irina}
}
source: https://github.com/rusnlp/rusnlp

Example usage on four threads, verbose:
    python pdf_to_text.py --source-dir ../../data/pdf/ --target-dir ../../data/extracted/txt -N 4 -v
"""

import argparse
import logging
from os import path, walk, makedirs, name
from io import StringIO
from multiprocessing import Pool, freeze_support, cpu_count
from pathlib import Path
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage

logging.getLogger("pdfminer3").setLevel(logging.ERROR)
errors = []

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=str, default='../../data/pdf')
    parser.add_argument('--target-dir', type=str,
                        default='../../data/extracted/txt')
    parser.add_argument('--error-dir', type=str, default='../../data/log')
    parser.add_argument('-N', type=int, default=4)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()
    return args


def convert_pdf_to_txt(filepath, page_range=(0, 0)):
    rm = PDFResourceManager()
    sio = StringIO()
    device = TextConverter(rm, sio, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(rm, device)
    with open(filepath, 'rb') as fp:
        pages = PDFPage.get_pages(fp=fp, pagenos=set(), maxpages=page_range[1],
                                  password='', caching=True,
                                  check_extractable=True)
        for page in pages:
            interpreter.process_page(page)
    text = sio.getvalue()
    device.close()
    sio.close()
    return text


def write_errors(error_dir, errors):
    if len(errors) > 0:
        with open(path.join(error_dir, "error.txt"), 'w', encoding='utf-8') as f:
            for error in errors:
                error_string = '{}'.format(error)
                logging.error(error_string)
                f.write(error_string + "\n")


def write_file(source_dir, saving_dir, root, file, text):
    with open(path.join(root.replace(source_dir, saving_dir), file).replace('.pdf', '.txt'), 'w',
              encoding='utf-8') as f:
        f.write(text)


def convert(source_dir, saving_dir, error_dir, N, page_range=(0, 0)):
    errors = []
    extracted_files = 0
    pool = Pool(N)
    for root, dirs, files in walk(source_dir):
        all_args = [(file, source_dir, saving_dir, error_dir, page_range, root)
                    for (i, file) in enumerate(files)]
        result = pool.map(convert_wrapper, all_args)
        extracted_files += sum(result)

    logging.info("Extracted {} files.".format(extracted_files))
    write_errors(error_dir, errors)


def convert_wrapper(args):
    file = args[0]
    source_dir = args[1]
    saving_dir = args[2]
    error_dir = args[3]
    page_range = args[4]
    root = args[5]

    if not file.endswith('pdf'):
        errors.append("Not a pdf file: {}".format(path.join(root, file)))
        return
    text = convert_pdf_to_txt(path.join(root, file), page_range)
    try:
        write_file(source_dir, saving_dir, root, file, text)
        logging.info("Extracted {} to {}".format(
            path.join(root, file),
            path.join(root.replace(source_dir, saving_dir), file).replace('.pdf', '.txt')))
    except FileNotFoundError:
        makedirs(path.dirname(path.join(root.replace(source_dir, saving_dir), file)))
        write_file(source_dir, saving_dir, root, file, text)
    return 1


def main() -> None:
    args = get_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    if args.verbose > 0:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    source_dir = str(Path(args.source_dir).resolve())
    saving_dir = str(Path(args.target_dir).resolve())
    error_dir = str(Path(args.error_dir).resolve())
    for p in [saving_dir, error_dir]:
        if not path.exists(p):
            makedirs(p)
            logging.info("Created directory: {}".format(p))

    # For multiprocessing on Windows
    if name == "nt":
        freeze_support()
    N = min(args.N, cpu_count())
    convert(source_dir, saving_dir, error_dir, N)


if __name__ == '__main__':
    main()
