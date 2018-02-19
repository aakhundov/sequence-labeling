import os
import pickle

import numpy as np


def load_embeddings(name, id_):
    if name.lower() == "polyglot":
        return load_polyglot(id_)
    elif name.lower() == "glove":
        return load_glove(id_)

    raise Exception("Unrecognized embeddings name: {}".format(name))


def load_polyglot(language):
    polyglot_file = "embeddings/polyglot-{}.pkl".format(language)
    words_file = "embeddings/polyglot-{}.words.txt".format(language)

    words, vecs = pickle.load(open(polyglot_file, "rb"), encoding="bytes")

    if not os.path.exists(words_file):
        with open(words_file, "w+", encoding="utf-8") as f:
            f.write("\n".join(words))

    return words_file, vecs, False


def load_glove(id_):
    glove_file = "embeddings/glove.{}.txt".format(id_)
    words_file = "embeddings/glove.{}.words.txt".format(id_.split(".")[0])
    vecs_file = "embeddings/glove.{}.vecs.npy".format(id_)

    if not os.path.exists(words_file) or not os.path.exists(vecs_file):
        words, vec_lists = [], []
        with open(glove_file, encoding="utf-8") as f:
            for line in [l[:-1] for l in f.readlines()]:
                tokens = line.split(" ")
                words.append(tokens[0])
                vec_lists.append([float(t) for t in tokens[1:]])
        with open(words_file, "w+", encoding="utf-8") as f:
            f.write("\n".join(["<UNK>"] + words))                        # adding <UNK> word at position 0
        vec_lists.insert(0, np.random.normal(size=[len(vec_lists[0])]))  # adding <UNK> vector at position 0
        np.save(vecs_file, np.array(vec_lists))

    vecs = np.load(vecs_file)
    uncased = id_.lower().startswith("6b")

    return words_file, vecs, uncased
