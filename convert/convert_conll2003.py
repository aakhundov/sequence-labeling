# This script converts CoNLL 2003 English and German "train", "testa",
# and "testb" files (https://www.clips.uantwerpen.be/conll2003/ner/) to
# the unified input format of the model (each line containing space-
# separated lists of tokens and labels of a single sentence, separated
# by a tab). The labeling scheme is converted to IOB2 (each entity
# starts with a B-tag). It is assumed that the six files - "eng.train",
# "eng.testa", "eng.testb", "deu.train", "deu.testa", and "deu.testb"
# (or just three files corresponding to one of the languages) are copied
# into --source-folder (-s). By default, the tags are converted to IOBES
# tagging scheme (this may be switched off by setting --iobes (-i) to
# False, to get IOB2 tags). The pre-processing results are written into
# two target folders: --target-folder-eng (-te) for English data and
# --target-folder-deu (-td) for German data, from where models can
# be trained directly using train.py.


import os
import argparse

import common


def fix_b_tag(pair, running_pairs):
    if pair[1].startswith("I-"):
        if len(running_pairs) == 0 or running_pairs[-1][1][2:] != pair[1][2:]:
            pair[1] = "B-" + pair[1][2:]
    return pair


def convert():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-folder", type=str, default="../data/sources/conll2003")
    parser.add_argument("-te", "--target-folder-eng", type=str, default="../data/ready/nerc/conll2003_eng")
    parser.add_argument("-tn", "--target-folder-deu", type=str, default="../data/ready/nerc/conll2003_deu")
    parser.add_argument("-i", "--iobes", type=bool, default=True)
    args = parser.parse_args()

    print("Source folder: {}".format(args.source_folder))
    print("Target folder (English): {}".format(args.target_folder_eng))
    print("Target folder (German): {}".format(args.target_folder_deu))
    print("Convert to IOBES: {}".format(args.iobes))
    print()

    args.target_folders = {
        "eng": args.target_folder_eng,
        "deu": args.target_folder_deu
    }

    for language in ["eng", "deu"]:
        sentence_pairs_per_file = {}

        for file in os.listdir(args.source_folder):
            if file.startswith(language):
                sentence_pairs_per_file[file] = []
                file_path = os.path.join(args.source_folder, file)
                file_lines = [l[:-1] for l in open(file_path, encoding="iso-8859-1").readlines()]

                print("processing data from {}".format(file_path))

                running_pairs = []
                for line in file_lines:
                    if line == "" or line.startswith("-DOCSTART-"):
                        if len(running_pairs) > 0:
                            sentence_pairs_per_file[file].append(
                                common.convert_to_iobes_tags(running_pairs)
                                if args.iobes else running_pairs
                            )
                            running_pairs = []
                        continue
                    tokens = line.split(" ")
                    pair = [tokens[0], tokens[-1]]
                    running_pairs.append(fix_b_tag(
                        pair, running_pairs
                    ))

        if len(sentence_pairs_per_file) == 0:
            print("{} files not found\n".format(language))
            continue

        if not os.path.exists(args.target_folders[language]):
            os.makedirs(args.target_folders[language])

        label_count_pairs = common.get_label_count_pairs(sentence_pairs_per_file)
        common.report_statistics(sentence_pairs_per_file, label_count_pairs)

        for target, source in [
            ["train", "{}.train".format(language)],
            ["val", "{}.testa".format(language)],
            ["test", "{}.testb".format(language)]
        ]:
            sentences_written, tokens_written = 0, 0
            out_path = os.path.join(args.target_folders[language], target + ".txt")

            with open(out_path, "w+", encoding="utf-8") as out:
                for sentence in sentence_pairs_per_file[source]:
                    out.write("{}\t{}\n".format(
                        " ".join([p[0] for p in sentence]),
                        " ".join([p[1] for p in sentence]),
                    ))
                    tokens_written += len(sentence)
                sentences_written += len(sentence_pairs_per_file[source])

            print("data from {} ({:,} sentences, {:,} tokens) written to {}".format(
                source, sentences_written, tokens_written, out_path
            ))

        label_path = os.path.join(args.target_folders[language], "labels.txt")
        with open(label_path, "w+", encoding="utf-8") as out:
            for lb in label_count_pairs:
                out.write("{}\n".format(lb[0]))

        print("{} labels written to {}".format(
            len(label_count_pairs), label_path
        ))
        print()


if __name__ == "__main__":
    convert()
