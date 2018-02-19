# This script converts CoNLL 2002 Spanish and Dutch "train", "testa",
# and "testb" files (https://www.clips.uantwerpen.be/conll2002/ner/)
# to the unified input format of the model (each line containing
# space-separated lists of tokens and labels of a single sentence,
# separated by a tab). It is assumed that the six files - "esp.train",
# "esp.testa", "eng.testb", "ned.train", "ned.testa", and "ned.testb"
# are copied into SOURCE_FOLDER. The pre-processing results
# are written into two TARGET_FOLDERS, from where a model may
# be trained directly using train.py.


import os


SOURCE_FOLDER = "../data/sources/conll2002"
TARGET_FOLDERS = {
    "esp": "../data/ready/nerc/conll2002_esp",
    "ned": "../data/ready/nerc/conll2002_ned"
}


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
    for language in ["esp", "ned"]:
        sentence_pairs_per_file = {}

        for file in os.listdir(SOURCE_FOLDER):
            if file.startswith(language):
                sentence_pairs_per_file[file] = []
                file_path = os.path.join(SOURCE_FOLDER, file)
                file_lines = [l[:-1] for l in open(file_path).readlines()]

                print("processing data from {}".format(file))

                running_pairs = []
                for line in file_lines:
                    if line == "" or line.startswith("-DOCSTART-"):
                        if len(running_pairs) > 0:
                            sentence_pairs_per_file[file].append(running_pairs)
                            running_pairs = []
                        continue
                    pair = line.split(" ")[0::2]
                    running_pairs.append(pair)

        if not os.path.exists(TARGET_FOLDERS[language]):
            os.mkdir(TARGET_FOLDERS[language])

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

        for target, source in [
            ["train", "{}.train".format(language)],
            ["val", "{}.testa".format(language)],
            ["test", "{}.testb".format(language)]
        ]:
            sentences_written, tokens_written = 0, 0
            out_path = os.path.join(TARGET_FOLDERS[language], target + ".txt")

            with open(out_path, "w+", encoding="utf-8") as out:
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

        label_path = os.path.join(TARGET_FOLDERS[language], "labels.txt")
        with open(label_path, "w+", encoding="utf-8") as out:
            for lb in label_count_pairs:
                out.write("{}\n".format(lb[0]))

        print("{} labels written to {}".format(
            len(label_count_pairs), label_path
        ))
        print()


if __name__ == "__main__":
    convert()