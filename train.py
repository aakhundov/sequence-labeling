import pickle

import tensorflow as tf

from model.input import input_fn
from model.model import model_fn
from model.metrics import compute_iob_metrics


EPOCHS = 100
BATCH_SIZE = 32

TASK_DATA_FOLDER = "data/nerc/"
POLYGLOT_FILE = "polyglot/polyglot-en.pkl"
REPORT_IOB_SCORES = True


print("Setting up input pipeline...")

with tf.device("/cpu:0"):
    train_data = input_fn(
        tf.data.TextLineDataset(TASK_DATA_FOLDER + "train.txt"),
        batch_size=BATCH_SIZE, shuffle_data=True
    )
    val_data = input_fn(tf.data.TextLineDataset(TASK_DATA_FOLDER + "val.txt"), shuffle_data=False)
    test_data = input_fn(tf.data.TextLineDataset(TASK_DATA_FOLDER + "test.txt"), shuffle_data=False)

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
next_input_values = iterator.get_next()

print("Loading embedding data...")

label_names = [line[:-1] for line in open(TASK_DATA_FOLDER + "labels.txt").readlines()]
polyglot_words, polyglot_embeddings = pickle.load(open(POLYGLOT_FILE, "rb"), encoding="bytes")

print("Building the model...")

train_op, loss, accuracy, predictions, labels, sentence_length, sentences, dropout_rate = model_fn(
    next_input_values, polyglot_words, polyglot_embeddings, label_names
)


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    print("Initializing variables...")

    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())

    train_init_op = iterator.make_initializer(train_data)
    val_init_op = iterator.make_initializer(val_data)
    test_init_op = iterator.make_initializer(test_data)

    print("Training...")
    print()

    for epoch in range(EPOCHS):
        sess.run(train_init_op)

        while True:
            try:
                sess.run(
                    train_op,
                    feed_dict={
                        dropout_rate: 0.5
                    }
                )
            except tf.errors.OutOfRangeError:
                break

        sess.run(val_init_op)
        val_acc, val_loss, val_labels, val_predictions, val_sentence_len = sess.run(
            [accuracy, loss, labels, predictions, sentence_length]
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
            print(epoch + 1, "val acc", val_acc)
