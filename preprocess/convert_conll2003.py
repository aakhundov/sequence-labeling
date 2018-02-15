# This script converts CoNLL 2003 English "train", "testa", and "testb"
# files (https://www.clips.uantwerpen.be/conll2003/ner/) to the unified
# input format of the model (each line containing space-separated lists
# of tokens and labels of a single sentence, separated by a tab). The
# labeling scheme is converted to IOB2 (each entity starts with a B-tag).
# It is assumed that the three files - "eng.train", "eng.testa", and
# "eng.testb" are copied into data/sources/conll2003 folder. The results
# of pre-processing are written into data/ready/nerc/conll2003 folder,
# from where a model may be trained directly using train.py.


import os


DUMMY_LABEL = "O"
SOURCE_FOLDER = "../data/sources/conll2003"
TARGET_FOLDER = "../data/ready/nerc/conll2003"


def fix_b_tag(pair, running_pairs):
    if pair[1].startswith("I-"):
        if len(running_pairs) == 0 or running_pairs[-1][1][2:] != pair[1][2:]:
            pair[1] = "B-" + pair[1][2:]
    return pair


def get_label_count_pairs(sentence_pairs_per_source):
    label_counts = {}
    for file in sentence_pairs_per_source.keys():
        for sentence in sentence_pairs_per_source[file]:
            for pair in sentence:
                label = pair[1]
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1

    labels = sorted(set(label_counts.keys()) - {DUMMY_LABEL}) + [DUMMY_LABEL]
    label_count_pairs = [(lb, label_counts[lb]) for lb in labels]

    return label_count_pairs


def convert():
    sentence_pairs_per_file = {}

    for file in os.listdir(SOURCE_FOLDER):
        sentence_pairs_per_file[file] = []
        file_path = os.path.join(SOURCE_FOLDER, file)
        file_lines = [l[:-1] for l in open(file_path).readlines()]

        print("processing data from {}".format(file_path))

        running_pairs = []
        for line in file_lines:
            if line == "" or line.startswith("-DOCSTART-"):
                if len(running_pairs) > 0:
                    sentence_pairs_per_file[file].append(running_pairs)
                    running_pairs = []
                continue

            pair = line.split(" ")[0::3]
            running_pairs.append(fix_b_tag(
                pair, running_pairs
            ))

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

    for target, source in [["train", "eng.train"], ["val", "eng.testa"], ["test", "eng.testb"]]:
        sentences_written, tokens_written = 0, 0
        out_path = os.path.join(TARGET_FOLDER, target + ".txt")

        with open(out_path, "w+") as out:
            for sentence in sentence_pairs_per_file[source]:
                out.write("{}\t{}\n".format(
                    " ".join([p[0] for p in sentence]),
                    " ".join([p[1] for p in sentence]),
                ))
                tokens_written += len(sentence)
            sentences_written += len(sentence_pairs_per_file[source])

        print("data from {} ({} sentences, {} tokens) written to {}".format(
            source, sentences_written, tokens_written, out_path
        ))

    label_path = os.path.join(TARGET_FOLDER, "labels.txt")
    with open(label_path, "w+") as out:
        for lb in label_count_pairs:
            out.write("{}\n".format(lb[0]))

    print("{} labels (with dummy \"{}\" in the end) written to {}".format(
        len(label_count_pairs), DUMMY_LABEL, label_path
    ))


if __name__ == "__main__":
    convert()
