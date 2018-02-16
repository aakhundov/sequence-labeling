# This script converts CoNLL 2000 English "train" and "test" files
# (https://www.clips.uantwerpen.be/conll2000/chunking/) to the unified
# input format of the model (each line containing space-separated lists
# of tokens and labels of a single sentence, separated by a tab).
# It is assumed that the two original files - "train.txt" and "test.txt"
# are copied into SOURCE_FOLDER. The data from "train.txt" is shuffled
# with a fixed seed, and split into training and validation set in 90/10
# proportion. The pre-processing results are written into TARGET_FOLDER,
# from where a model may be trained directly using train.py.


import os
import random


SOURCE_FOLDER = "../data/sources/conll2000"
TARGET_FOLDER = "../data/ready/chunk/conll2000"


def get_label_count_pairs(sentence_pairs_per_source):
    label_counts = {}
    for file in sentence_pairs_per_source.keys():
        for sentence in sentence_pairs_per_source[file]:
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

    train_bound = int(len(data) * 0.9)

    train = data[:train_bound]
    val = data[train_bound:]

    return train, val


def convert():
    sentence_pairs_per_file = {}

    for file in os.listdir(SOURCE_FOLDER):
        sentence_pairs_per_file[file] = []
        file_path = os.path.join(SOURCE_FOLDER, file)
        file_lines = [l[:-1] for l in open(file_path).readlines()]

        print("processing data from {}".format(file))

        running_pairs = []
        for line in file_lines:
            if line == "":
                if len(running_pairs) > 0:
                    sentence_pairs_per_file[file].append(running_pairs)
                    running_pairs = []
                continue
            pair = line.split(" ")[0::2]
            if len(pair) < 2:
                print(file, pair)
            running_pairs.append(pair)

    if not os.path.exists(TARGET_FOLDER):
        os.mkdir(TARGET_FOLDER)

    label_count_pairs = get_label_count_pairs(sentence_pairs_per_file)

    print()
    print("total sentences: {}\ntotal tokens: {}".format(
        sum(len(v) for v in sentence_pairs_per_file.values()),
        sum((sum(len(s) for s in v) for v in sentence_pairs_per_file.values()))
    ))
    print()
    print("labels with occurrence counts:")
    print(label_count_pairs)
    print()

    train, val = shuffle_and_split(sentence_pairs_per_file["train.txt"])
    test = sentence_pairs_per_file["test.txt"]

    for target, dataset in zip(["train", "val", "test"], [train, val, test]):
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
