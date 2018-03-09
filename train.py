import os
import time
import shutil
import argparse

import numpy as np
import tensorflow as tf

from model.input import input_fn
from model.model import model_fn
from util.embeddings import load_embeddings
from util.metrics import compute_metrics, get_class_f1_summary
from util.metrics import get_performance_summary, visualize_predictions
from util.misc import fetch_in_batches


def echo(log, *messages):
    print(*messages)
    joined = " ".join([str(m) for m in messages])
    log.write(joined + "\n")
    log.flush()


def create_training_artifacts(data_folder):
    if not os.path.exists("results"):
        os.mkdir("results")

    results_folder = "results/" + data_folder.replace("/", "_") + time.strftime("%Y%m%d_%H%M%S")
    results_folder = results_folder.replace("data_ready_", "").replace("data_", "")
    model_folder = os.path.join(results_folder, "model/")
    source_folder = os.path.join(results_folder, "source/")

    os.mkdir(results_folder)
    os.mkdir(model_folder)
    os.mkdir(source_folder)

    for folder in ["./", "model/", "util/"]:
        destination = source_folder
        if folder != "./":
            destination += folder
            os.makedirs(destination)
        for source_file in [f for f in os.listdir(folder) if f.endswith(".py")]:
            shutil.copy(folder + source_file, destination + source_file)

    log_file = open(os.path.join(results_folder, "log.txt"), "w+", encoding="utf-8")
    model_path = os.path.join(model_folder, "nlp-model")

    return model_path, log_file


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-folder", type=str, default="data/ready/nerc/conll2012")
    parser.add_argument("-em", "--embeddings-name", type=str, default="glove")
    parser.add_argument("-emid", "--embeddings-id", type=str, default="6B.100d")
    parser.add_argument("-ep", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-eb", "--eval-batch-size", type=int, default=2000)
    parser.add_argument("-lr", "--initial-learning-rate", type=float, default=0.001)
    parser.add_argument("-lrd", "--lr-decay-rate", type=float, default=0.05)
    parser.add_argument("-blu", "--byte-lstm-units", type=int, default=64)
    parser.add_argument("-wlu", "--word-lstm-units", type=int, default=128)
    parser.add_argument("-bed", "--byte-embedding-dim", type=int, default=50)
    parser.add_argument("-bll", "--byte-lstm-layers", type=int, default=1)
    parser.add_argument("-wll", "--word-lstm-layers", type=int, default=1)
    parser.add_argument("-be", "--use-byte-embeddings", type=int, default=1)
    parser.add_argument("-crf", "--use-crf-layer", type=int, default=1)
    args = parser.parse_args()

    assert os.path.exists(args.data_folder)
    if not args.data_folder.endswith("/"):
        args.data_folder += "/"

    print("Loading embeddings data...")
    emb_words, emb_vectors, uncased_embeddings = load_embeddings(args.embeddings_name, args.embeddings_id)
    label_names = [line[:-1] for line in open(args.data_folder + "labels.txt", encoding="utf-8").readlines()]

    print("Setting up input pipeline...")
    with tf.device("/cpu:0"):
        train_data = input_fn(
            tf.data.TextLineDataset(args.data_folder + "train.txt"),
            batch_size=args.batch_size, lower_case_words=uncased_embeddings,
            shuffle=True, cache=True, repeat=True
        )
        train_eval_data = input_fn(
            tf.data.TextLineDataset(args.data_folder + "train.txt"),
            batch_size=args.eval_batch_size, lower_case_words=uncased_embeddings,
            shuffle=False, cache=True, repeat=True
        )
        val_data = input_fn(
            tf.data.TextLineDataset(args.data_folder + "val.txt"),
            batch_size=args.eval_batch_size, lower_case_words=uncased_embeddings,
            shuffle=False, cache=True, repeat=True
        )

        train_data_count = sum(1 for _ in open(args.data_folder + "train.txt", encoding="utf-8"))
        val_data_count = sum(1 for _ in open(args.data_folder + "val.txt", encoding="utf-8"))

        data_handle = tf.placeholder(tf.string, shape=())
        next_input_values = tf.data.Iterator.from_string_handle(
            data_handle, train_data.output_types, train_data.output_shapes
        ).get_next()

    print("Building the model...")
    emb_words_placeholder = tf.placeholder(tf.string, [len(emb_words)])
    emb_vectors_placeholder = tf.placeholder(tf.float32, emb_vectors.shape)
    train_op, loss, accuracy, predictions, labels, \
        sentence_length, sentences, dropout_rate, completed_epochs = model_fn(
            input_values=next_input_values, label_vocab=label_names,
            embedding_words=emb_words_placeholder, embedding_vectors=emb_vectors_placeholder,
            byte_lstm_units=args.byte_lstm_units, word_lstm_units=args.word_lstm_units,
            byte_lstm_layers=args.byte_lstm_layers, word_lstm_layers=args.word_lstm_layers,
            byte_embedding_dim=args.byte_embedding_dim, training=True,
            initial_learning_rate=args.initial_learning_rate, lr_decay_rate=args.lr_decay_rate,
            use_byte_embeddings=bool(args.use_byte_embeddings),
            use_crf_layer=bool(args.use_crf_layer)
        )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print("Initializing variables...")
        sess.run(tf.tables_initializer(), feed_dict={emb_words_placeholder: emb_words})
        sess.run(tf.global_variables_initializer(), feed_dict={emb_vectors_placeholder: emb_vectors})
        del emb_words, emb_vectors

        train_handle = sess.run(train_data.make_one_shot_iterator().string_handle())
        train_eval_handle = sess.run(train_eval_data.make_one_shot_iterator().string_handle())
        val_handle = sess.run(val_data.make_one_shot_iterator().string_handle())

        best_metric, best_epoch = -1, 0
        saver = tf.train.Saver([
            v for v in tf.global_variables()
            if "known_word_embeddings" not in v.name
        ])

        print("Creating training artifacts...")
        model_path, log = create_training_artifacts(args.data_folder)

        print("Training...")
        print()

        echo(log, "data folder: {}".format(args.data_folder))
        echo(log, "embeddings: {}, {}".format(args.embeddings_name, args.embeddings_id))
        echo(log, "epochs: {}".format(args.epochs))
        echo(log, "batch size: {}".format(args.batch_size))
        echo(log, "initial learning rate: {}".format(args.initial_learning_rate))
        echo(log, "l.r. decay rate: {}".format(args.lr_decay_rate))
        echo(log, "byte lstm units: {}".format(args.byte_lstm_units))
        echo(log, "word lstm units: {}".format(args.word_lstm_units))
        echo(log, "byte embedding dim: {}".format(args.byte_embedding_dim))
        echo(log, "byte lstm layers: {}".format(args.byte_lstm_layers))
        echo(log, "word lstm layers: {}".format(args.word_lstm_layers))
        echo(log, "use byte embeddings: {}".format(args.use_byte_embeddings))
        echo(log, "use crf layer: {}".format(args.use_crf_layer))
        echo(log)

        for epoch in range(args.epochs):
            for step in range(-(-train_data_count // args.batch_size)):
                try:
                    sess.run(train_op, feed_dict={
                        data_handle: train_handle,
                        completed_epochs: epoch,
                        dropout_rate: 0.5
                    })
                except Exception as ex:
                    print(ex)

            for set_name, set_handle, set_size in [
                ["train", train_eval_handle, train_data_count],
                ["val", val_handle, val_data_count]
            ]:
                eval_loss, eval_labels, eval_predictions, eval_sentence_len = fetch_in_batches(
                    sess, [loss, labels, predictions, sentence_length], set_size,
                    feed_dict={data_handle: set_handle, dropout_rate: 0.0}
                )

                eval_metrics = compute_metrics(eval_labels, eval_predictions, eval_sentence_len, label_names)
                eval_message, eval_key_metric = get_performance_summary(eval_metrics, len(label_names))

                echo(log, "{:<22} {}".format(
                    "{0}.{1:<8} L {2:.3f}".format(
                        epoch + 1, set_name, eval_loss
                    ), eval_message
                ))

            echo(log)

            if eval_key_metric > best_metric:
                best_epoch = epoch + 1
                best_metric = eval_key_metric
                saver.save(sess, model_path)

        saver.restore(sess, model_path)
        best_labels, best_predictions, best_sentence_len, best_sentences = fetch_in_batches(
            sess, [labels, predictions, sentence_length, sentences], val_data_count,
            feed_dict={data_handle: val_handle, dropout_rate: 0.0}
        )

        best_metrics = compute_metrics(best_labels, best_predictions, best_sentence_len, label_names)
        best_message, best_key_metric = get_performance_summary(best_metrics, len(label_names))
        best_class_summary = get_class_f1_summary(best_metrics, label_names)

        np.set_printoptions(threshold=np.nan, linewidth=1000)

        echo(log)
        echo(log, "Best epoch: {}".format(best_epoch))
        echo(log, "Best metric: {:.2f}".format(best_key_metric))
        echo(log)
        echo(log, "Confusion matrix:\n")
        echo(log, best_metrics["confusion"])
        echo(log)

        if best_class_summary != "":
            echo(log, "Per-class summary:\n")
            echo(log, best_class_summary)

        echo(log, "Predicted sentence samples:\n")
        echo(log, visualize_predictions(
            best_sentences, best_labels, best_predictions,
            best_sentence_len, label_names, 100
        ))


if __name__ == "__main__":
    train()
