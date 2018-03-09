import os
import argparse

import tensorflow as tf

from model.input import input_fn
from model.model import model_fn
from util.embeddings import load_embeddings
from util.misc import fetch_in_batches, read_params_from_log


def annotate():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True)
    parser.add_argument("-r", "--results-folder", type=str, required=True)
    parser.add_argument("-o", "--output-file", type=str, default="")
    parser.add_argument("-b", "--batch-size", type=int, default=2000)
    args = parser.parse_args()

    assert os.path.exists(args.input_file)
    assert os.path.exists(args.results_folder)

    if args.output_file == "":
        args.output_file = os.path.splitext(args.input_file)[0] + ".labeled.txt"

    print("Input file: {}".format(args.input_file))
    print("Output file: {}".format(args.output_file))
    print("Results folder: {}".format(args.results_folder))
    print("Batch size: {}".format(args.batch_size))
    print()

    params = read_params_from_log(os.path.join(args.results_folder, "log.txt"))

    data_folder = params["data folder"]
    embeddings_name, embeddings_id = params["embeddings"].split(", ")
    char_lstm_units = int(params["char lstm units"]) if "char lstm units" in params else 64
    word_lstm_units = int(params["word lstm units"]) if "word lstm units" in params else 128
    char_embedding_dim = int(params["char embedding dim"]) if "char embedding dim" in params else 50
    char_lstm_layers = int(params["char lstm layers"]) if "char lstm layers" in params else 1
    word_lstm_layers = int(params["word lstm layers"]) if "word lstm layers" in params else 1
    use_char_embeddings = int(params["use char embeddings"]) if "use char embeddings" in params else 1
    use_crf_layer = int(params["use crf layer"]) if "use crf layer" in params else 1

    label_file = os.path.join(data_folder, "labels.txt")
    input_count = sum(1 for _ in open(args.input_file, encoding="utf-8"))

    print("Loading embeddings data...")
    emb_words, emb_vectors, uncased_embeddings = load_embeddings(embeddings_name, embeddings_id)
    label_names = [line[:-1] for line in open(label_file, encoding="utf-8").readlines()]

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
        char_lstm_units=char_lstm_units, word_lstm_units=word_lstm_units,
        char_lstm_layers=char_lstm_layers, word_lstm_layers=word_lstm_layers,
        char_embedding_dim=char_embedding_dim,
        use_char_embeddings=bool(use_char_embeddings),
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

        with open(args.output_file, "w+", encoding="utf-8") as out:
            for i in range(len(a_predictions)):
                out.write("{}\n".format(
                    " ".join([
                        label_names[lb] for lb in
                        a_predictions[i][:a_sentence_len[i]]
                    ])
                ))

        print()
        print("Results written to {}".format(args.output_file))


if __name__ == "__main__":
    annotate()
