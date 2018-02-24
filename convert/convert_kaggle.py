# This script converts "Annotated Corpus for Named Entity Recognition" available
# from Kaggle (https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data)
# to the unified input format of the model (each line containing space-separated
# lists of tokens and labels of a single sentence, separated by a tab). It is
# assumed that the file "ner_dataset.csv" containing the whole dataset is copied
# into --source-folder (-s). The data set is shuffled with a fixed seed, and split
# into training, validation, and test sets in 80/10/10 proportion. By default,
# the tags are converted to IOBES tagging scheme (this may be switched off by
# setting --iobes (-i) to False, to get IOB2 tags).The pre-processing results
# are written into --target-folder (-t), from where a model can be trained
# directly using train.py.


import os
import csv
import argparse

import common


def convert():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-folder", type=str, default="../data/sources/kaggle")
    parser.add_argument("-t", "--target-folder", type=str, default="../data/ready/nerc/kaggle")
    parser.add_argument("-i", "--iobes", type=bool, default=True)
    args = parser.parse_args()

    print("Source folder: {}".format(args.source_folder))
    print("Target folder: {}".format(args.target_folder))
    print("Convert to IOBES: {}".format(args.iobes))
    print()

    sentence_pairs = []

    file_path = os.path.join(args.source_folder, "ner_dataset.csv")
    with open(file_path, encoding="iso-8859-1") as f:
        file_lines = [l[:-1] for l in f.readlines()]

    print("processing data from {}".format(file_path))

    running_pairs = []
    for tokens in csv.reader(file_lines[1:]):
        if tokens[0].startswith("Sentence:") and len(running_pairs) > 0:
            sentence_pairs.append(
                common.convert_to_iobes_tags(running_pairs)
                if args.iobes else running_pairs
            )
            running_pairs = []
        running_pairs.append(tokens[1::2])
    if len(running_pairs) > 0:
        sentence_pairs.append(running_pairs)

    if not os.path.exists(args.target_folder):
        os.makedirs(args.target_folder)

    label_count_pairs = common.get_label_count_pairs(sentence_pairs)
    common.report_statistics(sentence_pairs, label_count_pairs)

    for target, dataset in zip(
        ["train", "val", "test"],
        common.shuffle_and_split(
            sentence_pairs, split_points=(0.8, 0.9)
        )
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
