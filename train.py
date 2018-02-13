import os
import time
import pickle
import shutil

import numpy as np
import tensorflow as tf

from model.input import input_fn
from model.model import model_fn
from model.metrics import compute_metrics


PHASES = 100
BATCH_SIZE = 8
STEPS_PER_PHASE = 1000

TASK_DATA_FOLDER = "data/nerc/"
POLYGLOT_FILE = "polyglot/polyglot-en.pkl"


def report_performance(metrics, num_labels):
    if metrics["F1"]:
        if metrics["IOB"]:
            return "{:<39} {:<30} {:<30} {}".format(
                "acc [B {d[B_acc]:.2f} E {d[E_acc]:.2f} O {d[O_acc]:.2f} T {d[acc]:.2f}]".format(d=metrics),
                "B [P {d[B_prec]:.2f} R {d[B_rec]:.2f} F1 {d[B_F1]:.2f}]".format(d=metrics),
                "E [P {d[E_prec]:.2f} R {d[E_rec]:.2f} F1 {d[E_F1]:.2f}]".format(d=metrics),
                "EC [P {d[EC_prec]:.2f} R {d[EC_rec]:.2f} F1 {d[EC_F1]:.2f}]".format(d=metrics)
            ), metrics["EC_F1"]
        else:
            if num_labels > 2:
                return "{:<30} {:<30} {}".format(
                    "acc [E {d[E_acc]:.2f} O {d[O_acc]:.2f} T {d[acc]:.2f}]".format(d=metrics),
                    "E [P {d[E_prec]:.2f} R {d[E_rec]:.2f} F1 {d[E_F1]:.2f}]".format(d=metrics),
                    "EC [P {d[EC_prec]:.2f} R {d[EC_rec]:.2f} F1 {d[EC_F1]:.2f}]".format(d=metrics)
                ), metrics["EC_F1"]
            else:
                return "{:<30} {}".format(
                    "acc [E {d[E_acc]:.2f} O {d[O_acc]:.2f} T {d[acc]:.2f}]".format(d=metrics),
                    "E [P {d[E_prec]:.2f} R {d[E_rec]:.2f} F1 {d[E_F1]:.2f}]".format(d=metrics)
                ), metrics["E_F1"]
    else:
        return "acc {d[acc]:.2f}".format(d=metrics), metrics["acc"]


def create_training_artifacts():
    if not os.path.exists("results"):
        os.mkdir("results")

    results_folder = "results/" + TASK_DATA_FOLDER.replace("/", "_") + time.strftime("%Y%m%d%H%M%S")
    model_folder, source_folder = os.path.join(results_folder, "model/"), os.path.join(results_folder, "source/")

    os.mkdir(results_folder)
    os.mkdir(model_folder)
    os.mkdir(source_folder)

    for folder in ["./", "model/"]:
        destination = source_folder
        if folder != "./":
            destination += folder
            os.makedirs(destination)
        for file in [f for f in os.listdir(folder) if f.endswith(".py")]:
            shutil.copy(folder + file, destination + file)

    log_file = open(os.path.join(results_folder, "log.txt"), "w+")
    model_path = os.path.join(model_folder, "nlp-model")

    return model_path, log_file


def echo(log, *messages):
    print(*messages)
    joined = " ".join([str(m) for m in messages])
    log.write(joined + "\n")
    log.flush()


def train():
    print("Creating training artifacts...")
    model_path, log = create_training_artifacts()

    print("Setting up input pipeline...")
    with tf.device("/cpu:0"):
        train_data = input_fn(
            tf.data.TextLineDataset(TASK_DATA_FOLDER + "train.txt"),
            batch_size=BATCH_SIZE, shuffle_data=True
        )
        val_data = input_fn(tf.data.TextLineDataset(TASK_DATA_FOLDER + "val.txt"))
        # test_data = input_fn(tf.data.TextLineDataset(TASK_DATA_FOLDER + "test.txt"))

        data_handle = tf.placeholder(tf.string, shape=())
        next_input_values = tf.data.Iterator.from_string_handle(
            data_handle, train_data.output_types, train_data.output_shapes
        ).get_next()

    print("Loading embedding data...")
    label_names = [line[:-1] for line in open(TASK_DATA_FOLDER + "labels.txt").readlines()]
    polyglot_words, polyglot_embeddings = pickle.load(open(POLYGLOT_FILE, "rb"), encoding="bytes")

    print("Building the model...")
    train_op, loss, accuracy, predictions, labels, sentence_length, sentences, dropout_rate = model_fn(
        next_input_values, polyglot_words, polyglot_embeddings, label_names,
        char_lstm_units=64, word_lstm_units=128, char_embedding_dim=50,
        char_lstm_layers=1, word_lstm_layers=1
    )

    with tf.Session() as sess:
        print("Initializing variables...")
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())

        train_handle = sess.run(train_data.make_one_shot_iterator().string_handle())
        val_handle = sess.run(val_data.make_one_shot_iterator().string_handle())
        # test_handle = sess.run(test_data.make_one_shot_iterator().string_handle())

        saver = tf.train.Saver()
        best_metric, best_phase = 0, 0

        print("Training...")
        print()

        for phase in range(PHASES):
            for step in range(STEPS_PER_PHASE):
                sess.run(train_op, feed_dict={data_handle: train_handle, dropout_rate: 0.5})

            val_loss, val_labels, val_predictions, val_sentence_len = sess.run(
                [loss, labels, predictions, sentence_length],
                feed_dict={data_handle: val_handle}
            )

            val_metrics = compute_metrics(val_labels, val_predictions, val_sentence_len, label_names)
            message, key_metric = report_performance(val_metrics, len(label_names))

            echo(log, "{:<17} {}".format("{0}.val: L {1:.3f}".format(phase+1, val_loss), message))

            if key_metric > best_metric:
                best_phase = phase + 1
                best_metric = key_metric
                saver.save(sess, model_path)

        echo(log)
        echo(log, "Best phase:", best_phase)
        echo(log, "Best metric:", best_metric)
        echo(log)

        saver.restore(sess, model_path)
        best_labels, best_predictions, best_sentence_len = sess.run(
            [labels, predictions, sentence_length],
            feed_dict={data_handle: val_handle}
        )

        best_metrics = compute_metrics(best_labels, best_predictions, best_sentence_len, label_names)

        np.set_printoptions(threshold=np.nan, linewidth=1000)
        echo(log, best_metrics["confusion"])


if __name__ == "__main__":
    train()
