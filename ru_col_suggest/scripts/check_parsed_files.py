"""Compares files of specified types from two specified directories. The intent
is to check that all pdf files were parsed into a file of a different type.

Example usage for pdf to txt and html, verbose:
    python check_parsed_files.py --source-dir ../../data/pdf
        --target-dir ../../data/extracted/txt --file-type .txt -v

    python check_parsed_files.py --source-dir ../../data/pdf
        --target-dir ../../data/extracted/html --file-type .html -v
"""

import argparse
import logging
import os


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=str, default='../../data/pdf')
    parser.add_argument('--target-dir', type=str,
                        default='../../data/extracted/')
    parser.add_argument('--file-type', type=str, default='.txt')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()
    return args


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
                full_path = os.path.join(path, file)
                rel_path = os.path.relpath(full_path, root)
                file_list.append(os.path.splitext(rel_path)[0])
    return file_list


def main() -> None:
    args = get_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    if args.verbose > 0:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    file_type = args.file_type
    source_files = sorted(get_all_files(args.source_dir, ".pdf"))
    target_files = sorted(get_all_files(args.target_dir, file_type))

    # Get the file deltas between the source and target directories
    delta_files = []
    idx_l = 0
    idx_r = 0
    len_l = len(source_files)
    len_r = len(target_files)

    while True:
        if idx_l == len_l or idx_r == len_r:
            break
        if source_files[idx_l] == target_files[idx_r]:
            idx_l += 1
            idx_r += 1
        elif source_files[idx_l] < target_files[idx_r]:
            delta_files.append(str(source_files[idx_l]) + ".pdf")
            idx_l += 1
        else:
            delta_files.append(str(target_files[idx_r]) + file_type)
            idx_r += 1

    # Check that there is at least one element in both lists
    delta_files.extend([str(file) + ".pdf" for file in source_files[idx_l:]])
    delta_files.extend([str(file) + file_type for file in target_files[idx_r:]])

    # Print the results
    if len(delta_files) == 0:
        logging.info("Passed!")
    for file in delta_files:
        logging.error("Unmatched file: {}".format(file))


if __name__ == '__main__':
    main()
