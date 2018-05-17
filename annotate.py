import os
import argparse

import tensorflow as tf

from model.input import input_fn
from model.model import model_fn
from util.embeddings import load_embeddings
from util.misc import fetch_in_batches, read_params_from_log
from util.metrics import are_iob_labels, are_iobes_labels


def iobes_to_iob(iobes_labels):
    iob_labels = []
    for label in iobes_labels:
        if label.startswith("S-"):
            iob_labels.append("B-" + label[2:])
        elif label.startswith("E-"):
            iob_labels.append("I-" + label[2:])
        else:
            iob_labels.append(label)
    return iob_labels


def fix_initial_i_tags(labels):
    num_fixed = 0
    fixed_labels = []
    for i in range(len(labels)):
        label = labels[i]
        if label.startswith("I-") and (i == 0 or labels[i-1][2:] != label[2:]):
            fixed_labels.append("B-" + label[2:])
            num_fixed += 1
        else:
            fixed_labels.append(label)
    return fixed_labels, num_fixed


def annotate():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True)
    parser.add_argument("-r", "--results-folder", type=str, required=True)
    parser.add_argument("-o", "--output-file", type=str, default="")
    parser.add_argument("-b", "--batch-size", type=int, default=2000)
    parser.add_argument("-iob", "--convert-to-iob", type=int, default=0)
    parser.add_argument("-fix", "--fix-i-tags", type=int, default=0)
    args = parser.parse_args()

    assert os.path.exists(args.input_file)
    assert os.path.exists(args.results_folder)

    if args.output_file == "":
        args.output_file = os.path.splitext(args.input_file)[0] + ".labeled.txt"

    print("Input file: {}".format(args.input_file))
    print("Output file: {}".format(args.output_file))
    print("Results folder: {}".format(args.results_folder))
    print("Batch size: {}".format(args.batch_size))
    print("Convert to IOB tags: {}".format(args.convert_to_iob))
    print("Fix initial I-tags: {}".format(args.fix_i_tags))
    print()

    params = read_params_from_log(os.path.join(args.results_folder, "log.txt"))

    data_folder = params["data folder"]
    embeddings_name, embeddings_id = params["embeddings"].split(", ")
    byte_lstm_units = int(params["byte lstm units"]) if "byte lstm units" in params else 64
    word_lstm_units = int(params["word lstm units"]) if "word lstm units" in params else 128
    byte_projection_dim = int(params["byte projection dim"]) if "byte projection dim" in params else 50
    byte_lstm_layers = int(params["byte lstm layers"]) if "byte lstm layers" in params else 1
    word_lstm_layers = int(params["word lstm layers"]) if "word lstm layers" in params else 1
    use_byte_embeddings = int(params["use byte embeddings"]) if "use byte embeddings" in params else 1
    use_word_embeddings = int(params["use word embeddings"]) if "use word embeddings" in params else 1
    use_crf_layer = int(params["use crf layer"]) if "use crf layer" in params else 1

    label_file = os.path.join(data_folder, "labels.txt")
    input_count = sum(1 for _ in open(args.input_file, encoding="utf-8"))

    print("Loading embeddings data...")
    emb_words, emb_vectors, uncased_embeddings = load_embeddings(embeddings_name, embeddings_id)
    label_names = [line[:-1] for line in open(label_file, encoding="utf-8").readlines()]

    iob_labels = are_iob_labels(label_names)
    iobes_labels = are_iobes_labels(label_names)

    if args.fix_i_tags and not (iob_labels or (iobes_labels and args.convert_to_iob)):
        raise Exception("Initial I-tags can be fixed only within IOB tagging scheme.")

    print("Setting up input pipeline...")
    with tf.device("/cpu:0"):
        next_input_values = input_fn(
            tf.data.TextLineDataset(args.input_file),
            batch_size=args.batch_size, lower_case_words=uncased_embeddings,
            shuffle=False, cache=False, repeat=False
        ).make_one_shot_iterator().get_next()

    print("Building the model...")
    emb_words_placeholder = tf.placeholder(tf.string, [len(emb_words)])
    emb_vectors_placeholder = tf.placeholder(tf.float32, emb_vectors.shape)
    _, _, _, predictions, _, sentence_length, _, _, _ = model_fn(
        input_values=next_input_values, label_vocab=label_names,
        embedding_words=emb_words_placeholder, embedding_vectors=emb_vectors_placeholder,
        byte_lstm_units=byte_lstm_units, word_lstm_units=word_lstm_units,
        byte_lstm_layers=byte_lstm_layers, word_lstm_layers=word_lstm_layers,
        byte_projection_dim=byte_projection_dim,
        use_byte_embeddings=bool(use_byte_embeddings),
        use_word_embeddings=bool(use_word_embeddings),
        use_crf_layer=bool(use_crf_layer)
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print("Initializing variables...")
        sess.run(tf.tables_initializer(), feed_dict={emb_words_placeholder: emb_words})
        sess.run(tf.global_variables_initializer(), feed_dict={emb_vectors_placeholder: emb_vectors})
        del emb_words, emb_vectors

        tf.train.Saver([
            v for v in tf.global_variables()
            if "known_word_embeddings" not in v.name
        ]).restore(sess, os.path.join(
            args.results_folder, "model", "nlp-model"
        ))

        print("Annotating...")
        print()

        a_predictions, a_sentence_len = fetch_in_batches(
            sess, [predictions, sentence_length], total=input_count,
            progress_callback=lambda fetched: print("{} / {} done".format(
                fetched, input_count
            ))
        )

        total_fixed = 0
        with open(args.output_file, "w+", encoding="utf-8") as out:
            for i in range(len(a_predictions)):
                labels = [label_names[lb] for lb in a_predictions[i][:a_sentence_len[i]]]

                if iobes_labels or iob_labels:
                    if iobes_labels and args.convert_to_iob:
                        labels = iobes_to_iob(labels)
                    if (iob_labels or (iobes_labels and args.convert_to_iob)) and args.fix_i_tags:
                        labels, num_fixed = fix_initial_i_tags(labels)
                        total_fixed += num_fixed

                out.write("{}\n".format(" ".join(labels)))

        print()
        print("Results written to {}".format(args.output_file))

        if args.fix_i_tags:
            print("({} initial I-tags were fixed)".format(total_fixed))


if __name__ == "__main__":
    annotate()
