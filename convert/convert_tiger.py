# This script converts the sentences and POS-labels from German Tiger corpus
# (http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger.en.html)
# to the unified input format of the model (each line containing space-separated
# lists of tokens and labels of a single sentence, separated by a tab). It is
# assumed that a single *.xml file containing the whole set of structured data
# (e.g. "tiger_release_aug07.corrected.16012013.xml") is copied into SOURCE_FOLDER.
# In case if --source-folder (-s) contains multiple *.xml files, only the first
# found *.xml file from is processed. The data set is shuffled with a fixed seed,
# and split into training, validation, and test sets in 80/10/10 proportion. The
# pre-processing results are written into --target-folder (-t), from where a
# model can be trained directly using train.py.


import os
import re
import random
import argparse

import xml.etree.ElementTree


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
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-folder", type=str, default="../data/sources/tiger")
    parser.add_argument("-t", "--target-folder", type=str, default="../data/ready/pos/tiger")
    args = parser.parse_args()

    print("Source folder: {}".format(args.source_folder))
    print("Target folder: {}".format(args.target_folder))
    print()

    sentence_pairs = []

    for file in os.listdir(args.source_folder):
        if re.match(".*\.xml$", file):
            print("processing data from {}".format(file))

            file_path = os.path.join(args.source_folder, file)
            root = xml.etree.ElementTree.parse(file_path).getroot()
            sentence_tags = root.find("body").findall("s")

            for sentence_tag in sentence_tags:
                word_tags = sentence_tag.find("graph").find("terminals")
                word_pairs = [[w.attrib["word"], w.attrib["pos"]] for w in word_tags]
                sentence_pairs.append(word_pairs)

            break

    if not os.path.exists(args.target_folder):
        os.makedirs(args.target_folder)

    label_count_pairs = get_label_count_pairs(sentence_pairs)

    print()
    print("total sentences: {:,}\ntotal tokens: {:,}".format(
        len(sentence_pairs), sum(len(s) for s in sentence_pairs)
    ))
    print()
    print("labels with occurrence counts:")
    print([(lb, "{:,}".format(lbc)) for lb, lbc in label_count_pairs])
    print()

    for target, dataset in zip(
            ["train", "val", "test"],
            shuffle_and_split(sentence_pairs)
    ):
        sentences_written, tokens_written = 0, 0
        out_path = os.path.join(args.target_folder, target + ".txt")

        with open(out_path, "w+", encoding="utf-8") as out:
            for sentence in dataset:
                out.write("{}\t{}\n".format(
                    " ".join([p[0] for p in sentence]),
                    " ".join([p[1] for p in sentence]),
                ))
                tokens_written += len(sentence)
            sentences_written = len(dataset)

        print("{:,} sentences ({:,} tokens) written to {}".format(
            sentences_written, tokens_written, out_path
        ))

    label_path = os.path.join(args.target_folder, "labels.txt")
    with open(label_path, "w+", encoding="utf-8") as out:
        for lb in label_count_pairs:
            out.write("{}\n".format(lb[0]))

    print("{} labels written to {}".format(
        len(label_count_pairs), label_path
    ))


if __name__ == "__main__":
    convert()
