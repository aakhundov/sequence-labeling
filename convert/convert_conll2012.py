# This script converts CoNLL 2012 English training, development, and testing data
# (http://conll.cemantix.org/2012/data.html) to the unified input format of the model
# (each line containing space-separated lists of tokens and labels of a single sentence,
# separated by a tab). It is assumed that the three folders - "train", "development",
# and "test" - from the "data" folder in the original corpus containing "*.gold_conll"
# files with resolved tokens ([WORD] placeholders substituted by real words) are copied
# into --source-folder (-s). Data for three different tasks - POS (Part-Of-Speech
# Tagging), NERC (Named Entity Recognition and Classification), and PRED (PREdicate
# Detection) - is extracted from the corpus simultaneously and written to three
# separate target folders: --target-folder-pos (-tp), --target-folder-nerc (-tn),
# and --target-folder-pred (-tpr), from where models can be trained directly using
# train.py. By default, the NERC tags are converted to IOBES tagging scheme (this
# may be switched off by setting --iobes (-i) to False, to get IOB2 tags).


import os
import re
import argparse

import common


def enumerate_files(folder, pattern=""):
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        if not os.path.isdir(path):
            if re.search(pattern, entry):
                yield path
        else:
            yield from enumerate_files(
                path, pattern
            )


def fix_token(t):
    if "B-" in t:
        t = t.replace("-LRB-", "(")
        t = t.replace("-RRB-", ")")
        t = t.replace("-LCB-", "{")
        t = t.replace("-RCB-", "}")
        t = t.replace("-LSB-", "[")
        t = t.replace("-RSB-", "]")
    if t.startswith("/"):
        if t == "/.":
            t = "."
        if t == "/?":
            t = "?"
        if t == "/-":
            t = "-"
    return t


def decode_nerc_labels(labels):
    result = []
    current = ""
    for lb in labels:
        if lb.startswith("("):
            result.append("B-" + lb[1:-1])
            if not lb.endswith(")"):
                current = lb[1:-1]
        else:
            result.append(
                "I-" + current
                if current != ""
                else "O"
            )
            if lb.endswith(")"):
                current = ""
    return result


def split_by_task(pairs):
    sentence = [fix_token(p[0]) for p in pairs]
    return [
        ["POS", list(zip(sentence, [fix_token(p[1]["POS"]) for p in pairs]))],
        ["NERC", list(zip(sentence, decode_nerc_labels([p[1]["NERC"] for p in pairs])))],
        ["PRED", list(zip(sentence, [p[1]["PRED"] for p in pairs]))]
    ]


def get_joint_pair(line):
    tokens = list(filter(None, line.split(" ")))
    pair = [tokens[3], {
        "POS": tokens[4], "NERC": tokens[10],
        "PRED": "V" if "(V*)" in tokens[11:-1] else "-"
    }]
    return pair


def convert():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-folder", type=str, default="../data/sources/conll2012")
    parser.add_argument("-tp", "--target-folder-pos", type=str, default="../data/ready/pos/conll2012")
    parser.add_argument("-tn", "--target-folder-nerc", type=str, default="../data/ready/nerc/conll2012")
    parser.add_argument("-tpr", "--target-folder-pred", type=str, default="../data/ready/pred/conll2012")
    parser.add_argument("-i", "--iobes", type=bool, default=True)
    args = parser.parse_args()

    print("Source folder: {}".format(args.source_folder))
    print("Target folder for POS task: {}".format(args.target_folder_pos))
    print("Target folder for NERC task: {}".format(args.target_folder_nerc))
    print("Target folder for PRED task: {}".format(args.target_folder_pred))
    print("Convert to IOBES: {}".format(args.iobes))
    print()

    args.target_folders = {
        "POS": args.target_folder_pos,
        "NERC": args.target_folder_nerc,
        "PRED": args.target_folder_pred
    }

    sentence_pairs_per_task_and_folder = {
        t: {} for t in args.target_folders.keys()
    }

    for folder in ["train", "development", "test"]:
        folder_path = os.path.join(args.source_folder, folder)
        for task in sentence_pairs_per_task_and_folder.keys():
            sentence_pairs_per_task_and_folder[task][folder] = []

        print("processing data from {} folder".format(folder_path))

        for path in enumerate_files(folder_path, "\.gold_conll$"):
            file_pairs = {t: [] for t in args.target_folders.keys()}
            with open(path, encoding="utf-8") as f:
                running_joint_pairs = []
                for line in [l[:-1] for l in f.readlines()]:
                    if line == "" or line.startswith("#"):
                        if len(running_joint_pairs) > 0:
                            for task, pairs in split_by_task(running_joint_pairs):
                                # excluding sentences with rare labels from POS data
                                if task == "POS" and any(p[1] in ["*", "AFX"] for p in pairs):
                                    continue
                                file_pairs[task].append(
                                    common.convert_to_iobes_tags(pairs)
                                    if task == "NERC" and args.iobes else pairs
                                )
                            running_joint_pairs = []
                        continue
                    running_joint_pairs.append(
                        get_joint_pair(line)
                    )

            # excluding files with only "XX" or "VERB" POS labels from POS data
            if any(any(p[1] not in ["XX", "VERB"] for p in s) for s in file_pairs["POS"]):
                sentence_pairs_per_task_and_folder["POS"][folder].extend(file_pairs["POS"])
            # excluding files without named entity labelling from NERC data
            if any(any(p[1] != "O" for p in s) for s in file_pairs["NERC"]):
                sentence_pairs_per_task_and_folder["NERC"][folder].extend(file_pairs["NERC"])
            # excluding files without predicate labelling from PRED data
            if any(any(p[1] == "V" for p in s) for s in file_pairs["PRED"]):
                sentence_pairs_per_task_and_folder["PRED"][folder].extend(file_pairs["PRED"])

    for task in ["POS", "NERC", "PRED"]:
        print("\n--------------------------------------\n")
        print("Data for {} task:".format(task))

        target_folder = args.target_folders[task]
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        dummy = "-" if task == "PRED" else None
        label_count_pairs = common.get_label_count_pairs(sentence_pairs_per_task_and_folder[task], dummy)
        common.report_statistics(sentence_pairs_per_task_and_folder[task], label_count_pairs)

        for target, source in [["train", "train"], ["val", "development"], ["test", "test"]]:
            sentences_written, tokens_written = 0, 0
            out_path = os.path.join(target_folder, target + ".txt")

            with open(out_path, "w+", encoding="utf-8") as out:
                for sentence in sentence_pairs_per_task_and_folder[task][source]:
                    out.write("{}\t{}\n".format(
                        " ".join([p[0] for p in sentence]),
                        " ".join([p[1] for p in sentence]),
                    ))
                    tokens_written += len(sentence)
                sentences_written += len(sentence_pairs_per_task_and_folder[task][source])

            print("data from {} folder ({:,} sentences, {:,} tokens) written to {}".format(
                source, sentences_written, tokens_written, out_path
            ))

        label_path = os.path.join(target_folder, "labels.txt")
        with open(label_path, "w+", encoding="utf-8") as out:
            for lb in label_count_pairs:
                out.write("{}\n".format(lb[0]))

        print("{} labels written to {}".format(
            len(label_count_pairs), label_path
        ))


if __name__ == "__main__":
    convert()
