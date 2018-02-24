import random


def are_iob_labels(label_names):
    return all(lb[:2] in ["B-", "I-"] or lb == "O" for lb in label_names)


def are_iobes_labels(label_names):
    return all(lb[:2] in ["B-", "I-", "E-", "S-"] or lb == "O" for lb in label_names) and \
           any(lb[:2] in ["E-", "S-"] for lb in label_names)


def complete_label_counts(label_counts):
    if are_iobes_labels(label_counts.keys()):
        if are_iobes_labels(label_counts.keys()):
            prefixes = ["B", "I", "E", "S"]
        else:
            prefixes = ["B", "I"]

        classes = set(e[2:] for e in label_counts.keys() if e.startswith("B-"))

        for class_ in classes:
            for prefix in prefixes:
                tag = "{}-{}".format(prefix, class_)
                if tag not in label_counts:
                    label_counts[tag] = 0

    return label_counts


def convert_to_iobes_tags(sentence):
    result = []
    for i in range(len(sentence)):
        if i >= len(sentence)-1 or not sentence[i+1][1].startswith("I-"):
            if sentence[i][1].startswith("B-"):
                result.append((sentence[i][0], "S-" + sentence[i][1][2:]))
            elif sentence[i][1].startswith("I-"):
                result.append((sentence[i][0], "E-" + sentence[i][1][2:]))
            else:
                result.append(sentence[i])
        else:
            result.append(sentence[i])
    return result


def get_all_sentences(sentence_pairs_collection):
    return sum(sentence_pairs_collection.values(), []) \
        if isinstance(sentence_pairs_collection, dict) \
        else sentence_pairs_collection


def get_label_count_pairs(sentence_pairs_collection, dummy_label=None):
    label_counts = {}
    for sentence in get_all_sentences(sentence_pairs_collection):
        for pair in sentence:
            label = pair[1]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

    label_counts = complete_label_counts(label_counts)
    labels = sorted(label_counts.keys())

    if dummy_label is None and (are_iob_labels(labels) or are_iobes_labels(labels)):
        dummy_label = "O"
    if dummy_label is not None:
        labels = sorted(set(labels) - {dummy_label}) + [dummy_label]

    return [(lb, label_counts[lb]) for lb in labels]


def report_statistics(sentence_collection, label_count_pairs):
    all_sentences = get_all_sentences(sentence_collection)
    max_label_len = max(len(lb[0]) for lb in label_count_pairs)

    print()
    print("total sentences: {:,}\ntotal tokens: {:,}".format(
        len(all_sentences), sum(len(s) for s in all_sentences)
    ))
    print()
    print("labels with occurrence counts:\n")
    for i in range(len(label_count_pairs)):
        lp = label_count_pairs[i]
        print(("{:<" + str(max_label_len + 15) + "}").format(
            "{}:  {:,}".format(lp[0], lp[1])
        ), end="")
        if (i + 1) % 4 == 0 or i == len(label_count_pairs) - 1:
            print()
    print()


def shuffle_and_split(data, split_points=(), seed=12345):
    random.seed(seed)
    random.shuffle(data)
    random.seed()

    split = [int(len(data) * s) for s in [0] + list(split_points)]

    result = []
    for i in range(len(split)-1):
        result.append(data[split[i]:split[i+1]])
    result.append(data[split[-1]:])

    return result
