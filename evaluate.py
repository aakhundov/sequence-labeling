import os
import re
import sys

import numpy as np
import tensorflow as tf

from model.input import input_fn
from model.model import model_fn
from util.embeddings import load_embeddings
from util.metrics import compute_metrics, get_class_f1_summary
from util.metrics import get_performance_summary, visualize_predictions
from util.misc import fetch_in_batches


def evaluate():
    results_folder = sys.argv[1]
    data_file = sys.argv[2] if len(sys.argv) > 2 else "val.txt"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    num_to_visualize = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    with open(os.path.join(results_folder, "log.txt"), encoding="utf-8") as f:
        data_folder = re.split(":\s+", f.readline()[:-1])[1]
        embeddings_name, embeddings_id = re.split(":\s+", f.readline()[:-1])[1].split(", ")

    label_file = os.path.join(data_folder, "labels.txt")
    data_file = os.path.join(data_folder, data_file)
    data_count = sum(1 for _ in open(data_file))

    print("Loading embeddings data...")
    embedding_words, embedding_vectors, uncased_embeddings = load_embeddings(embeddings_name, embeddings_id)
    label_names = [line[:-1] for line in open(label_file, encoding="utf-8").readlines()]

    print("Setting up input pipeline...")
    with tf.device("/cpu:0"):
        next_input_values = input_fn(
            tf.data.TextLineDataset(data_file),
            batch_size=batch_size, lower_case_words=uncased_embeddings,
            shuffle=False, cache=False, repeat=False
        ).make_one_shot_iterator().get_next()

    print("Building the model...")
    embedding_words_placeholder = tf.placeholder(tf.string, [len(embedding_words)])
    embedding_vectors_placeholder = tf.placeholder(tf.float32, embedding_vectors.shape)
    _, loss, _, predictions, labels, sentence_length, sentences, _ = model_fn(
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

        print("Evaluating...")
        print()

        e_loss, e_predictions, e_labels, e_sentence_len, e_sentences = fetch_in_batches(
            sess, [loss, predictions, labels, sentence_length, sentences], total=data_count,
            progress_callback=lambda fetched: print("{} / {} done".format(fetched, data_count))
        )

        e_metrics = compute_metrics(e_labels, e_predictions, e_sentence_len, label_names)
        e_message, e_key_metric = get_performance_summary(e_metrics, len(label_names))
        e_class_summary = get_class_f1_summary(e_metrics, label_names)

        np.set_printoptions(threshold=np.nan, linewidth=1000)

        print()
        print("Loss: {:.3f}".format(e_loss))
        print("Key metric: {:.2f}".format(e_key_metric))
        print()
        print("Performance summary:\n")
        print(e_message)
        print()
        print("Confusion matrix:\n")
        print(e_metrics["confusion"])
        print()

        if e_class_summary != "":
            print("Per-class summary:\n")
            print(e_class_summary)

        if num_to_visualize > 0:
            print("Predicted sentence samples:\n")
            print(visualize_predictions(
                e_sentences, e_labels, e_predictions,
                e_sentence_len, label_names, num_to_visualize
            ))


if __name__ == "__main__":
    evaluate()
