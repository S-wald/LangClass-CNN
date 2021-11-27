import os
import re

import numpy as np

LETTER = 'abcdefghijklmnopqrstuvwxyz'
DIGITS = '0123456789'
OTHERS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
ALPHABET = LETTER + DIGITS + OTHERS
PAD_VECTOR = [0 for x in ALPHABET]


def get_one_hot_enc_mapping():
    # pre-calculated one-hot vectors:
    supported_chars_map = {}
    for i, ch in enumerate(ALPHABET):
        vec = [0 for x in ALPHABET]
        vec[i] = 1
        supported_chars_map[ch] = vec
    return supported_chars_map


SUPPORTED_CHARS_MAP = get_one_hot_enc_mapping()

LANGS = [
    "C", "C#", "C++",
    "D", "Haskell",
    "Java", "JavaScript",
    "PHP", "Python",
    "Rust"
]

NUM_CLASSES = len(LANGS)


def get_input_and_labels(root_folder, sample_vectors_size=1024, breakup=True):
    X = []
    Y = []
    FILES = []
    for i, lang in enumerate(LANGS):
        print('Processing language:', lang)
        # One-hot class label vector:
        class_label = [0 for x in range(0, NUM_CLASSES)]
        class_label[i] = 1
        # For all files in language folder:
        folder = os.path.join(root_folder, lang)
        for fn in os.listdir(folder):
            if fn.startswith("."):
                continue  # Skip hidden files and Jupyterlab cache directories
            file_name = os.path.join(folder, fn)
            sample_vectors = file_to_vectors(file_name,
                                             sample_vectors_size=sample_vectors_size,
                                             breakup=breakup)
            for fv in sample_vectors:
                X.append(fv)  # the sample feature vector
                Y.append(class_label)  # the class ground-truth
                FILES.append(file_name)

    return np.array(X, dtype=np.int8), np.array(Y, dtype=np.int8), FILES


def file_to_vectors(file_name, sample_vectors_size=1024,
                    normalize_whitespace=True, breakup=True):
    samples = get_source_snippets(file_name, breakup)
    return [sample_to_vector(s, sample_vectors_size, normalize_whitespace)
            for s in samples]


def sample_to_vector(sample, sample_vectors_size=1024,
                     normalize_whitespace=True):
    if normalize_whitespace:
        # Map (most) white-space to space and compact to single one:
        sample = sample.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        sample = re.sub(f'\s+', ' ', sample)

    # Encode the characters to one-hot vectors:
    sample_vectors = []
    for ch in sample:
        if ch in SUPPORTED_CHARS_MAP:
            sample_vectors.append(SUPPORTED_CHARS_MAP[ch])

    # Truncate to fixed length:
    sample_vectors = sample_vectors[0:sample_vectors_size]

    # Pad with 0 vectors:
    # all-zeroes padding vector:
    if len(sample_vectors) < sample_vectors_size:
        for i in range(0, sample_vectors_size - len(sample_vectors)):
            sample_vectors.append(PAD_VECTOR)

    return np.array(sample_vectors)


def get_source_snippets(file_name, breakup=True):
    # Read the file content and lower-case:
    text = ""
    with open(file_name, mode='r', encoding="utf8") as file:
        text = file.read().lower()
    lines = text.split('\n')
    nlines = len(lines)
    if breakup and nlines > 50:
        aThird = nlines // 3
        twoThirds = 2 * aThird
        text1 = '\n'.join(lines[:aThird])
        text2 = '\n'.join(lines[aThird:twoThirds])
        text3 = '\n'.join(lines[twoThirds:])
        return [text1, text2, text3]
    return [text]
