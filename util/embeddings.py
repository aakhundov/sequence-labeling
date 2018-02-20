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
    vectors_file = "data/embeddings/glove/glove.{}.npy".format(id_)

    words, vectors = None, None

    if not os.path.exists(words_file) or not cache:
        print("Loading GloVe words...")

        words = ["<UNK>"]  # adding <UNK> word at position 0
        with open(glove_file, encoding="utf-8") as file:
            for line in file:
                words.append(line[:line.find(" ")])

        if cache:
            with open(words_file, "w+", encoding="utf-8") as f:
                f.write("\n".join(words))

    if words is None:
        with open(words_file, encoding="utf-8") as file:
            words = [l[:-1] for l in file.readlines()]

    if not os.path.exists(vectors_file) or not cache:
        print("Loading GloVe vectors...")

        count, dim = len(words), int(id_.split(".")[-1][:-1])
        vectors = np.zeros([count, dim], dtype=np.float32)

        # adding random vector for <UNK> at position 0
        vectors[0] = np.random.normal(size=[dim], loc=0.0, scale=0.4)

        cursor = 1
        with open(glove_file, encoding="utf-8") as file:
            for line in file:
                vectors[cursor] = np.fromstring(
                    line[line.find(" ")+1:],
                    dtype=np.float32, count=dim, sep=" "
                )
                if cursor % 100000 == 0:
                    print("{} vectors loaded".format(cursor))
                cursor += 1

        if cache:
            np.save(vectors_file, vectors)

    if vectors is None:
        vectors = np.load(vectors_file)

    uncased = not id_.lower().startswith("840b")

    return words, vectors, uncased
