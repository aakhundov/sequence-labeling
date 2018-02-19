import os
import sys
import time
import shutil

import numpy as np
import tensorflow as tf

from model.input import input_fn
from model.model import model_fn
from util.embeddings import load_embeddings
from util.metrics import compute_metrics, get_class_f1_summary
from util.metrics import get_performance_summary, visualize_predictions


PHASES = 100
BATCH_SIZE = 8
STEPS_PER_PHASE = 1000

DEFAULT_DATA_FOLDER = "data/ready/pos/wsj/"
DEFAULT_EMBEDDINGS_NAME = "glove"
DEFAULT_EMBEDDINGS_ID = "6B.100d"


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
    data_folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_FOLDER
    embeddings_name = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_EMBEDDINGS_NAME
    embeddings_id = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_EMBEDDINGS_ID

    if not data_folder.endswith("/"):
        data_folder += "/"

    print("Loading embeddings data...")
    embedding_words, embedding_vectors, uncased_embeddings = load_embeddings(embeddings_name, embeddings_id)
    label_names = [line[:-1] for line in open(data_folder + "labels.txt", encoding="utf-8").readlines()]

    print("Setting up input pipeline...")
    with tf.device("/cpu:0"):
        train_data = input_fn(
            tf.data.TextLineDataset(data_folder + "train.txt"),
            batch_size=BATCH_SIZE, lower_case_words=uncased_embeddings,
            shuffle=True, cache=True, repeat=True
        )
        val_data = input_fn(
            tf.data.TextLineDataset(data_folder + "val.txt"),
            batch_size=None, lower_case_words=uncased_embeddings,
            shuffle=False, cache=True, repeat=True
        )

        data_handle = tf.placeholder(tf.string, shape=())
        next_input_values = tf.data.Iterator.from_string_handle(
            data_handle, train_data.output_types, train_data.output_shapes
        ).get_next()

    print("Building the model...")
    embedding_words_placeholder = tf.placeholder(tf.string, [len(embedding_words)])
    embedding_vectors_placeholder = tf.placeholder(tf.float32, embedding_vectors.shape)
    train_op, loss, accuracy, predictions, labels, sentence_length, sentences, dropout_rate = model_fn(
        next_input_values, embedding_words_placeholder, embedding_vectors_placeholder, label_names, training=True,
        char_lstm_units=64, word_lstm_units=128, char_embedding_dim=50,
        char_lstm_layers=1, word_lstm_layers=1
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print("Initializing variables...")
        sess.run(tf.tables_initializer(), feed_dict={
            embedding_words_placeholder: embedding_words
        })
        sess.run(tf.global_variables_initializer(), feed_dict={
            embedding_vectors_placeholder: embedding_vectors
        })

        train_handle = sess.run(train_data.make_one_shot_iterator().string_handle())
        val_handle = sess.run(val_data.make_one_shot_iterator().string_handle())

        best_metric, best_phase = -1, 0
        saver = tf.train.Saver([
            v for v in tf.global_variables()
            if "known_word_embeddings" not in v.name
        ])

        print("Creating training artifacts...")
        model_path, log = create_training_artifacts(data_folder)

        print("Training...")
        print()

        echo(log, "data folder:  {}".format(data_folder))
        echo(log, "embeddings:   {}, {}".format(embeddings_name, embeddings_id))
        echo(log)

        for phase in range(PHASES):
            for step in range(STEPS_PER_PHASE):
                try:
                    sess.run(train_op, feed_dict={data_handle: train_handle, dropout_rate: 0.5})
                except Exception as ex:
                    print(ex)

            val_loss, val_labels, val_predictions, val_sentence_len = sess.run(
                [loss, labels, predictions, sentence_length],
                feed_dict={data_handle: val_handle}
            )

            val_metrics = compute_metrics(val_labels, val_predictions, val_sentence_len, label_names)
            message, key_metric = get_performance_summary(val_metrics, len(label_names))

            echo(log, "{:<17} {}".format("{0}.val: L {1:.3f}".format(phase+1, val_loss), message))

            if key_metric > best_metric:
                best_phase = phase + 1
                best_metric = key_metric
                saver.save(sess, model_path)

        saver.restore(sess, model_path)
        best_labels, best_predictions, best_sentence_len, best_sentences = sess.run(
            [labels, predictions, sentence_length, sentences],
            feed_dict={data_handle: val_handle}
        )

        best_metrics = compute_metrics(best_labels, best_predictions, best_sentence_len, label_names)
        best_message, best_key_metric = get_performance_summary(best_metrics, len(label_names))
        class_summary = get_class_f1_summary(best_metrics, label_names)

        np.set_printoptions(threshold=np.nan, linewidth=1000)

        echo(log)
        echo(log, "Best phase: {}".format(best_phase))
        echo(log, "Best metric: {:.2f}".format(best_key_metric))
        echo(log)
        echo(log, "Confusion matrix:\n")
        echo(log, best_metrics["confusion"])
        echo(log)

        if class_summary != "":
            echo(log, "Per-class summaries:\n")
            echo(log, class_summary)

        echo(log, "Predicted sentence samples:\n")
        echo(log, visualize_predictions(
            best_sentences, best_labels, best_predictions,
            best_sentence_len, label_names, 100
        ))


if __name__ == "__main__":
    train()
