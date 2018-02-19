import os
import pickle

import numpy as np


def load_embeddings(name, id_, cache=True):
    if name.lower() == "polyglot":
        return load_polyglot(id_, cache)
    elif name.lower() == "glove":
        return load_glove(id_, cache)

    raise Exception("Unrecognized embeddings name: {}".format(name))


def load_polyglot(language, *_):
    polyglot_file = "data/embeddings/polyglot/polyglot-{}.pkl".format(language)
    words, vectors = pickle.load(open(polyglot_file, "rb"), encoding="bytes")
    return words, vectors, False


def load_glove(id_, cache):
    glove_file = "data/embeddings/glove/glove.{}.txt".format(id_)
    words_file = "data/embeddings/glove/glove.{}.txt".format(id_.split(".")[0])
    vecs_file = "data/embeddings/glove/glove.{}.npy".format(id_)

    words, vectors = None, None

    if not os.path.exists(words_file) or not cache:
        words = ["<UNK>"]  # adding <UNK> word at position 0
        with open(glove_file, encoding="utf-8") as f:
            for line in [l[:-1] for l in f.readlines()]:
                words.append(line[:line.find(" ")])
        if cache:
            with open(words_file, "w+", encoding="utf-8") as f:
                f.write("\n".join(words))

    if not os.path.exists(vecs_file) or not cache:
        vector_lists = []
        with open(glove_file, encoding="utf-8") as f:
            for line in [l[:-1] for l in f.readlines()]:
                vector_lists.append([float(t) for t in line.split(" ")[1:]])
        # adding one random normal vector for <UNK> words at position 0
        vector_lists.insert(0, np.random.normal(size=[len(vector_lists[0])]))
        vectors = np.array(vector_lists)
        if cache:
            np.save(vecs_file, vectors)

    if words is None:
        with open(words_file, encoding="utf-8") as wf:
            words = [l[:-1] for l in wf.readlines()]
    if vectors is None:
        vectors = np.load(vecs_file)

    return words, vectors, id_.lower().startswith("6b")
