import pickle
from datetime import datetime

import tensorflow as tf

from model.input import input_fn
from model.model import model_fn
from model.metrics import compute_iob_metrics


EPOCHS = 100
BATCH_SIZE = 32
STEPS_PER_EPOCH = 1000

TASK_DATA_FOLDER = "data/nerc/"
POLYGLOT_FILE = "polyglot/polyglot-en.pkl"
REPORT_IOB_SCORES = True


def echo(*msgs):
    print("[{}] {}".format(
        datetime.now().strftime('%H:%M:%S.%f')[:-3],
        " ".join([str(m) for m in msgs])
    ))


echo("Preparing input...")

with tf.device("/cpu:0"):
    train_data = input_fn(
        tf.data.TextLineDataset(TASK_DATA_FOLDER + "train.txt"),
        batch_size=BATCH_SIZE, shuffle_data=True
    )
    val_data = input_fn(tf.data.TextLineDataset(TASK_DATA_FOLDER + "val.txt"), shuffle_data=False)
    test_data = input_fn(tf.data.TextLineDataset(TASK_DATA_FOLDER + "test.txt"), shuffle_data=False)

data_handle = tf.placeholder(tf.string, shape=())
next_input_values = tf.data.Iterator.from_string_handle(
    data_handle, train_data.output_types, train_data.output_shapes
).get_next()


echo("Building the model...")

label_names = [line[:-1] for line in open(TASK_DATA_FOLDER + "labels.txt").readlines()]
polyglot_words, polyglot_embeddings = pickle.load(open(POLYGLOT_FILE, "rb"), encoding="bytes")

train_op, loss, accuracy, predictions, labels, sentence_length, sentences, dropout_rate = model_fn(
    next_input_values, polyglot_words, polyglot_embeddings, label_names
)


with tf.Session() as sess:
    echo("Initializing variables...")

    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())

    train_handle = sess.run(train_data.make_one_shot_iterator().string_handle())
    val_handle = sess.run(val_data.make_one_shot_iterator().string_handle())
    test_handle = sess.run(test_data.make_one_shot_iterator().string_handle())

    echo("Training...")
    print()

    for epoch in range(EPOCHS):
        for step in range(STEPS_PER_EPOCH):
            sess.run(
                train_op,
                feed_dict={
                    data_handle: train_handle,
                    dropout_rate: 0.5
                }
            )

        val_acc, val_loss, val_labels, val_predictions, val_sentence_len = sess.run(
            [accuracy, loss, labels, predictions, sentence_length],
            feed_dict={
                data_handle: val_handle
            }
        )

        if REPORT_IOB_SCORES:
            iob_metrics = compute_iob_metrics(
                val_labels, val_predictions, val_sentence_len, len(label_names)
            )

            print("{:<16} {:<39} {:<30} {:<30} {}".format(
                "{0}.val: L {1:.3f}".format(epoch + 1, val_loss),
                "acc [B {d[B_acc]:.2f} E {d[E_acc]:.2f} O {d[O_acc]:.2f} T {d[acc]:.2f}]".format(d=iob_metrics),
                "B [P {d[B_prec]:.2f} R {d[B_rec]:.2f} F1 {d[B_F1]:.2f}]".format(d=iob_metrics),
                "E [P {d[E_prec]:.2f} R {d[E_rec]:.2f} F1 {d[E_F1]:.2f}]".format(d=iob_metrics),
                "EC [P {d[EC_prec]:.2f} R {d[EC_rec]:.2f} F1 {d[EC_F1]:.2f}]".format(d=iob_metrics),
            ))
        else:
            echo(epoch + 1, "val acc", val_acc)