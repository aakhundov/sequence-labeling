import os
import argparse

import numpy as np
import tensorflow as tf

from model.input import input_fn
from model.model import model_fn
from util.embeddings import load_embeddings
from util.metrics import compute_metrics, get_class_f1_summary
from util.metrics import get_performance_summary, visualize_predictions
from util.misc import fetch_in_batches, read_params_from_log


def evaluate():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-r", metavar="results-folder", type=str, required=True,
        help="the path to the folder with the training results")
    parser.add_argument(
        "-f", metavar="data-file", type=str, default="val.txt",
        help="the file (within the original data folder) "
             "with the data to evaluate the model on")
    parser.add_argument(
        "-b", metavar="batch-size", type=int, default=2000,
        help="the batch size used for the evaluation")
    parser.add_argument(
        "-v", metavar="num-to-show", type=int, default=0,
        help="the number of predicted sentence samples to visualize")
    args = parser.parse_args()

    assert os.path.exists(args.results_folder)

    print("Results folder: {}".format(args.results_folder))
    print("Data file: {}".format(args.data_file))
    print("Batch size: {}".format(args.batch_size))
    print("Samples to show: {}".format(args.num_to_show))
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
    data_file = os.path.join(data_folder, args.data_file)
    data_count = sum(1 for _ in open(data_file, encoding="utf-8"))

    print("Loading embeddings data...")
    emb_words, emb_vectors, uncased_embeddings = load_embeddings(embeddings_name, embeddings_id)
    label_names = [line[:-1] for line in open(label_file, encoding="utf-8").readlines()]

    print("Setting up input pipeline...")
    with tf.device("/cpu:0"):
        next_input_values = input_fn(
            tf.data.TextLineDataset(data_file),
            batch_size=args.batch_size, lower_case_words=uncased_embeddings,
            shuffle=False, cache=False, repeat=False
        ).make_one_shot_iterator().get_next()

    print("Building the model...")
    emb_words_placeholder = tf.placeholder(tf.string, [len(emb_words)])
    emb_vectors_placeholder = tf.placeholder(tf.float32, emb_vectors.shape)
    _, loss, _, predictions, labels, sentence_length, sentences, _, _ = model_fn(
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

        if args.num_to_show > 0:
            print("Predicted sentence samples:\n")
            print(visualize_predictions(
                e_sentences, e_labels, e_predictions,
                e_sentence_len, label_names, args.num_to_show
            ))


if __name__ == "__main__":
    evaluate()
