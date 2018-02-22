import os
import re
import sys

import tensorflow as tf

from model.input import input_fn
from model.model import model_fn
from util.embeddings import load_embeddings
from util.misc import fetch_in_batches


def annotate():
    input_file, results_folder = sys.argv[1:3]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    with open(os.path.join(results_folder, "log.txt"), encoding="utf-8") as f:
        data_folder = re.split(":\s+", f.readline()[:-1])[1]
        embeddings_name, embeddings_id = re.split(":\s+", f.readline()[:-1])[1].split(", ")

    label_file = os.path.join(data_folder, "labels.txt")
    input_count = sum(1 for _ in open(input_file))

    print("Loading embeddings data...")
    embedding_words, embedding_vectors, uncased_embeddings = load_embeddings(embeddings_name, embeddings_id)
    label_names = [line[:-1] for line in open(label_file, encoding="utf-8").readlines()]

    print("Setting up input pipeline...")
    with tf.device("/cpu:0"):
        next_input_values = input_fn(
            tf.data.TextLineDataset(input_file),
            batch_size=batch_size, lower_case_words=uncased_embeddings,
            shuffle=False, cache=False, repeat=False
        ).make_one_shot_iterator().get_next()

    print("Building the model...")
    embedding_words_placeholder = tf.placeholder(tf.string, [len(embedding_words)])
    embedding_vectors_placeholder = tf.placeholder(tf.float32, embedding_vectors.shape)
    _, _, _, predictions, _, sentence_length, _, _ = model_fn(
        next_input_values, embedding_words_placeholder, embedding_vectors_placeholder, label_names, training=False,
        char_lstm_units=64, word_lstm_units=128, char_embedding_dim=50,
        char_lstm_layers=1, word_lstm_layers=1
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print("Initializing variables...")
        sess.run(tf.tables_initializer(), feed_dict={embedding_words_placeholder: embedding_words})
        sess.run(tf.global_variables_initializer(), feed_dict={embedding_vectors_placeholder: embedding_vectors})
        del embedding_words, embedding_vectors

        tf.train.Saver([
            v for v in tf.global_variables()
            if "known_word_embeddings" not in v.name
        ]).restore(sess, os.path.join(
            results_folder, "model", "nlp-model"
        ))

        print("Annotating...")
        print()

        a_predictions, a_sentence_len = fetch_in_batches(
            sess, [predictions, sentence_length], total=input_count,
            progress_callback=lambda fetched: print("{} / {} done".format(
                fetched, input_count
            ))
        )

        output_file = os.path.splitext(input_file)[0] + ".labeled.txt"
        with open(output_file, "w+", encoding="utf-8") as out:
            for i in range(len(a_predictions)):
                out.write("{}\n".format(
                    " ".join([
                        label_names[lb] for lb in
                        a_predictions[i][:a_sentence_len[i]]
                    ])
                ))

        print()
        print("Results written to {}".format(output_file))


if __name__ == "__main__":
    annotate()
