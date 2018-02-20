# This script converts CoNLL 2012 English training, development, and
# testing data (http://conll.cemantix.org/2012/data.html) to the unified
# input format of the model (each line containing space-separated lists
# of tokens and labels of a single sentence, separated by a tab). It is
# assumed that the three folders - "train", "development", and "test" -
# from the "data" folder in the original corpus containing "*.gold_conll"
# files with resolved tokens ([WORD] placeholders substituted by real words)
# are copied into SOURCE_FOLDER. Data for three tasks - POS (Part-Of-Speech
# Tagging), NERC (Named Entity Recognition and Classification), and PRED
# (PREdicate Detection) - is extracted from the corpus simultaneously and
# written to separate TARGET_FOLDERS, from where models can be trained
# directly using train.py.


import os


SOURCE_FOLDER = "../data/sources/conll2012"
TARGET_FOLDERS = {
    "POS": "../data/ready/pos/conll2012",
    "NERC": "../data/ready/nerc/conll2012",
    "PRED": "../data/ready/pred/conll2012"
}


def fix_token(t):
    if "B-" in t:
        t = t.replace("-LRB-", "(")
        t = t.replace("-RRB-", ")")
        t = t.replace("-LCB-", "{")
        t = t.replace("-RCB-", "}")
        t = t.replace("-LSB-", "[")
        t = t.replace("-RSB-", "]")
    if t == "/.":
        t = "."
    if t == "/?":
        t = "?"
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
        "PRED": "V" if "(V*)" in tokens[11:-1] else "X"
    }]
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

    return [(lb, label_counts[lb]) for lb in sorted(label_counts.keys())]


def convert():
    sentence_pairs_per_task_and_folder = {
        task: {} for task in TARGET_FOLDERS.keys()
    }

    for folder in ["train", "development", "test"]:
        for task in sentence_pairs_per_task_and_folder.keys():
            sentence_pairs_per_task_and_folder[task][folder] = []

        print("processing data from {} folder".format(folder))

        for root, _, files in os.walk(os.path.join(SOURCE_FOLDER, folder)):
            for file in [f for f in files if f.endswith(".gold_conll")]:
                    with open(os.path.join(root, file), encoding="utf-8") as f:
                        running_joint_pairs = []
                        for line in [l[:-1] for l in f.readlines()]:
                            if line == "" or line.startswith("#"):
                                if len(running_joint_pairs) > 0:
                                    for task, pairs in split_by_task(running_joint_pairs):
                                        # excluding sentences with "XX" and rare labels from POS task
                                        if task == "POS" and any([p[1] in ["XX", "*", "AFX"] for p in pairs]):
                                            continue
                                        sentence_pairs_per_task_and_folder[task][folder].append(pairs)
                                    running_joint_pairs = []
                                continue
                            running_joint_pairs.append(
                                get_joint_pair(line)
                            )

    for task in ["POS", "NERC", "PRED"]:
        print("\n--------------------------------------\n")
        print("Data for {} task:\n".format(task))

        target_folder = TARGET_FOLDERS[task]
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

        label_count_pairs = get_label_count_pairs(sentence_pairs_per_task_and_folder[task])

        print("total sentences: {}\ntotal tokens: {}\n".format(
            sum(len(v) for v in sentence_pairs_per_task_and_folder[task].values()),
            sum((sum(len(s) for s in v) for v in sentence_pairs_per_task_and_folder[task].values()))
        ))
        print("labels with occurrence counts:")
        print(label_count_pairs)
        print()

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

            print("data from {} folder ({} sentences, {} tokens) written to {}".format(
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
