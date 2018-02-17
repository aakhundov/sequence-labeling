# This script converts "Annotated Corpus for Named Entity Recognition" available
# from Kaggle (https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data)
# to the unified input format of the model (each line containing space-separated
# lists of tokens and labels of a single sentence, separated by a tab). It is
# assumed that the file "ner_dataset.csv" containing the whole dataset is copied
# into SOURCE_FOLDER. The data set is shuffled with a fixed seed, and split into
# training, validation, and test sets in 80/10/10 proportion. The pre-processing
# results are written into TARGET_FOLDER, from where a model may be trained
# directly using train.py.


import os
import csv
import random


SOURCE_FOLDER = "../data/sources/kaggle"
TARGET_FOLDER = "../data/ready/nerc/kaggle"
DATASET_FILE = "ner_dataset.csv"


def get_label_count_pairs(sentence_pairs):
    label_counts = {}
    for sentence in sentence_pairs:
        for pair in sentence:
            label = pair[1]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

    return [(lb, label_counts[lb]) for lb in sorted(label_counts.keys())]


def shuffle_and_split(data):
    random.seed(12345)
    random.shuffle(data)
    random.seed()

    train_bound = int(len(data) * 0.8)
    val_bound = int(len(data) * 0.9)

    train = data[:train_bound]
    val = data[train_bound:val_bound]
    test = data[val_bound:]

    return train, val, test


def convert():
    sentence_pairs = []

    file_path = os.path.join(SOURCE_FOLDER, DATASET_FILE)
    with open(file_path, encoding="iso-8859-1") as f:
        file_lines = [l[:-1] for l in f.readlines()]

    print("processing data from {}".format(DATASET_FILE))

    running_pairs = []
    for tokens in csv.reader(file_lines[1:]):
        if tokens[0].startswith("Sentence:") and len(running_pairs) > 0:
            sentence_pairs.append(running_pairs)
            running_pairs = []
        running_pairs.append(tokens[1::2])
    if len(running_pairs) > 0:
        sentence_pairs.append(running_pairs)

    if not os.path.exists(TARGET_FOLDER):
        os.mkdir(TARGET_FOLDER)

    label_count_pairs = get_label_count_pairs(sentence_pairs)

    print()
    print("total sentences: {}\ntotal tokens: {}".format(
        len(sentence_pairs), sum(len(s) for s in sentence_pairs)
    ))
    print()
    print("labels with occurrence counts:")
    print(label_count_pairs)
    print()

    for target, dataset in zip(
            ["train", "val", "test"],
            shuffle_and_split(sentence_pairs)
    ):
        sentences_written, tokens_written = 0, 0
        out_path = os.path.join(TARGET_FOLDER, target + ".txt")

        with open(out_path, "w+", encoding="utf-8") as out:
            for sentence in dataset:
                out.write("{}\t{}\n".format(
                    " ".join([p[0] for p in sentence]),
                    " ".join([p[1] for p in sentence]),
                ))
                tokens_written += len(sentence)
            sentences_written = len(dataset)

        print("{} sentences ({} tokens) written to {}".format(
            sentences_written, tokens_written, out_path
        ))

    label_path = os.path.join(TARGET_FOLDER, "labels.txt")
    with open(label_path, "w+", encoding="utf-8") as out:
        for lb in label_count_pairs:
            out.write("{}\n".format(lb[0]))

    print("{} labels written to {}".format(
        len(label_count_pairs), label_path
    ))


if __name__ == "__main__":
    convert()
