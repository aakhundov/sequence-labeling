import pickle
import numpy as np


def load_embeddings(name, id_):
    if name.lower() == "polyglot":
        return load_polyglot(id_)
    elif name.lower() == "glove":
        return load_glove(id_)

    raise Exception("Unrecognized embeddings name: {}".format(name))


def load_polyglot(language):
    file = "embeddings/polyglot-{}.pkl".format(language)
    words, vectors = pickle.load(open(file, "rb"), encoding="bytes")
    return words, vectors, False


def load_glove(id_):
    words, vectors = [], []
    file = "embeddings/glove.{}.txt".format(id_)
    with open(file) as f:
        for line in [l[:-1] for l in f.readlines()]:
            tokens = line.split(" ")
            words.append(tokens[0])
            vectors.append([float(t) for t in tokens[1:]])
    return words, np.array(vectors), id_.lower().startswith("6b")
