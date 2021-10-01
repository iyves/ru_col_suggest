import os
import re
import argparse
import logging

"""Combine all preprocessed text () from the cybercat corpus into one file.
"""

## taken from check_parsed_files.py
def get_all_files(root: str, extension: str):
    """
    Returns the filenames of all files in the root directory that have the
    specified extension.

    :param root: The root directory to search.
    :param extension: The type of file extension to search for.
    :return: A list of relative paths to filenames.
    """
    file_list = []
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(path, file))
    return file_list


## taken from https://stackoverflow.com/questions/15203829/python-argparse-file-extension-checking
def CheckExt(choices):
    class Act(argparse.Action):
        def __call__(self, parser, namespace, tfile, option_string=None):
            ext = os.path.splitext(tfile)[1][1:]
            if ext not in choices:
                option_string = '({})'.format(option_string) if option_string else ''
                parser.error("File doesn't end with one of {}{}".format(choices, option_string))
            else:
                setattr(namespace, self.dest, tfile)
    return Act


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', help='source directory')
    parser.add_argument('--target-file', action=CheckExt({'txt', }), help='target .txt file')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    if args.verbose > 0:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    source_files = sorted(get_all_files(args.source_dir, ".txt"))
    target_filename = os.path.abspath(args.target_file)

    full_corpus_target_dir = os.path.dirname(target_filename)
    if not os.path.exists(full_corpus_target_dir):
        os.makedirs(full_corpus_target_dir)
    with open(target_filename, 'w', encoding='utf-8') as to_write:
        for i, file in enumerate(source_files, 1):
            if i % 1000 == 0:
                print(f'{i} files processed')
            for sentence in open(file, 'r', encoding='utf-8'):
                s = re.sub(' +', ' ', sentence)
                s = s.replace(" ", "").replace(" .", ".").replace(" ,", ",")
                lens = [len(word) for word in s.split() if word != "NUM_arab"]
                if len(lens) >= 5:
                    avg = sum(lens) / len(lens)
                    if avg > 5:
                        to_write.write(s)


if __name__ == '__main__':
    main()