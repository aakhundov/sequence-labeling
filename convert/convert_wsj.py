# This script converts Wall Street Journal (parsed + POS tags) part of
# the original Treebank-3 corpus (https://catalog.ldc.upenn.edu/ldc99t42)
# to the unified input format of the model (each line containing
# space-separated lists of tokens and labels of a single sentence,
# separated by a tab). It is assumed that the contents of parsed/mrg/wsj
# folder of Treebank-3 (25 folders from "00" to "24") are copied into
# SOURCE_FOLDER. The pre-processing results are written into
# TARGET_FOLDER, from where a model can be trained directly
# using train.py.


import os
import re


SOURCE_FOLDER = "../data/sources/wsj"
TARGET_FOLDER = "../data/ready/pos/wsj"


def fix_pair(p):
    if "\\" in p[0]:
        p[0] = p[0].replace("\\", "")
    if p[0].startswith("-"):
        p[0] = p[0].replace("-LRB-", "(")
        p[0] = p[0].replace("-RRB-", ")")
        p[0] = p[0].replace("-LCB-", "{")
        p[0] = p[0].replace("-RCB-", "}")
    if p[1].startswith("-"):
        p[1] = p[1].replace("-LRB-", "(")
        p[1] = p[1].replace("-RRB-", ")")
    return p


def tree_lines_to_pairs(tree):
    sentence = []
    for line in tree:
        for match in re.findall("\([^()]+\)", line.strip()):
            pair = match[1:-1].split(" ")
            if pair[0] != "-NONE-":
                pair = [pair[1], pair[0]]
                sentence.append(fix_pair(pair))
    return sentence


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


def convert():
    sentences_pairs_per_section = {}

    for folder in [f for f in sorted(os.listdir(SOURCE_FOLDER)) if re.match("\d{2}", f)]:
        section = int(folder)
        sentences_pairs_per_section[section] = []
        folder_path = os.path.join(SOURCE_FOLDER, folder)

        print("processing section {} from {}".format(section, folder_path))

        for file in [f for f in sorted(os.listdir(folder_path)) if re.match("wsj_\d{4}.mrg", f)]:
            file_path = os.path.join(SOURCE_FOLDER, folder, file)
            file_lines = [l[:-1] for l in open(file_path).readlines()]

            running_tree_lines = []
            for line in file_lines:
                if not line.startswith(" ") and len(running_tree_lines) > 0:
                    sentences_pairs_per_section[section].append(
                        tree_lines_to_pairs(running_tree_lines)
                    )
                    running_tree_lines = []
                if line != "":
                    running_tree_lines.append(line)
            sentences_pairs_per_section[section].append(
                tree_lines_to_pairs(running_tree_lines)
            )

    if not os.path.exists(TARGET_FOLDER):
        os.mkdir(TARGET_FOLDER)

    label_count_pairs = get_label_count_pairs(sentences_pairs_per_section)

    print()
    print("total sentences: {}\ntotal tokens: {}".format(
        sum(len(v) for v in sentences_pairs_per_section.values()),
        sum((sum(len(s) for s in v) for v in sentences_pairs_per_section.values()))
    ))
    print()
    print("labels with occurrence counts:")
    print(label_count_pairs)
    print()

    for target, from_, to in [["train", 0, 18], ["val", 19, 21], ["test", 22, 24]]:
        sentences_written, tokens_written = 0, 0
        out_path = os.path.join(TARGET_FOLDER, target + ".txt")

        with open(out_path, "w+", encoding="utf-8") as out:
            for section in range(from_, to+1):
                for sentence in sentences_pairs_per_section[section]:
                    out.write("{}\t{}\n".format(
                        " ".join([p[0] for p in sentence]),
                        " ".join([p[1] for p in sentence]),
                    ))
                    tokens_written += len(sentence)
                sentences_written += len(sentences_pairs_per_section[section])

        print("sections {}-{} ({} sentences, {} tokens) written to {}".format(
            from_, to, sentences_written, tokens_written, out_path
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
