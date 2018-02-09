import pickle

from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.crf as crf


EPOCHS = 100
BATCH_SIZE = 32
STEPS_PER_EPOCH = 1000

CHAR_EMBEDDING_DIM = 50
CHAR_LSTM_UNITS = 64
WORD_LSTM_UNITS = 128

TASK_DATA_FOLDER = "data/nerc/"
POLYGLOT_FILE = "polyglot/polyglot-en.pkl"
REPORT_IOB_F1_SCORE = True


def echo(*msgs):
    print("[{}] {}".format(
        datetime.now().strftime('%H:%M:%S.%f')[:-3],
        " ".join([str(m) for m in msgs])
    ))


def input_fn(input_lines, batch_size=None, shuffle_data=False):
    sentence_lines = input_lines.map(lambda x: tf.string_split([x], "\t").values[0], 4)
    sentences_tokens = sentence_lines.map(lambda x: tf.string_split([x], " ").values, 4)
    padded_sentences = sentences_tokens.map(lambda x: tf.concat((["<S>"], x, ["</S>"]), axis=0), 4)
    padded_sentence_length = padded_sentences.map(lambda x: tf.shape(x)[0])

    label_lines = input_lines.map(lambda x: tf.string_split([x], "\t").values[-1], 4)
    labels_tokens = label_lines.map(lambda x: tf.string_split([x], " ").values, 4)
    padded_labels = labels_tokens.map(lambda x: tf.concat(([""], x, [""]), axis=0), 4)

    words = padded_sentences.map(lambda x: tf.reshape(x, [-1, 1]), 4)
    padded_word_length = words.map(
        lambda x: tf.map_fn(lambda y: tf.size(tf.string_split(y, "")) + 2, x, dtype=tf.int32), 4
    )

    def decode_word(word, max_len):
        w_bytes = tf.concat(([1], tf.decode_raw(word[0], tf.uint8)[:max_len], [2]), axis=0)
        padded = tf.pad(w_bytes, [[0, max_len - tf.shape(w_bytes)[0]]])
        return padded

    max_byte_length = padded_word_length.map(lambda x: tf.reduce_max(x), 4)
    padded_word_bytes = tf.data.Dataset.zip((words, max_byte_length)).map(
        lambda x, y: tf.map_fn(lambda z: decode_word(z, y), x, dtype=tf.uint8), 4
    )

    data = tf.data.Dataset.zip((
        tf.data.Dataset.zip((sentence_lines, padded_sentences, padded_sentence_length)),
        tf.data.Dataset.zip((words, padded_word_length, padded_word_bytes)),
        padded_labels
    ))

    def make_padded_batches(d):
        return d.padded_batch(
            batch_size if batch_size is not None else 1000000000,
            (([], [-1], []), ([-1, -1], [-1], [-1, -1]), [-1]),
            (("", "", 0), ("", 0, tf.constant(0, tf.uint8)), "")
        )

    if batch_size is not None:
        data = data.cache()
        if shuffle_data:
            data = data.shuffle(batch_size * 256)
        return data.apply(make_padded_batches).repeat().prefetch(16)
    else:
        return data.apply(make_padded_batches).cache().repeat()


def get_char_input(char_tensor, embedding_dim):
    return tf.reshape(
        tf.feature_column.input_layer(
            {"chars": tf.reshape(char_tensor, [-1])},
            [tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    "chars", 256, default_value=0
                ), dimension=embedding_dim, trainable=True
            )]
        ),
        tf.concat((tf.shape(char_tensor), [embedding_dim]), axis=0)
    )


def get_word_input(word_tensor, embedding_words, embedding_matrix):
    _, embedding_dim = embedding_matrix.shape

    return tf.reshape(
        tf.feature_column.input_layer(
            {"sentences": tf.reshape(word_tensor, [-1])},
            [tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    "sentences", embedding_words, default_value=0
                ), dimension=embedding_dim, trainable=False,
                initializer=tf.initializers.constant(embedding_matrix)
            )]
        ),
        tf.concat((tf.shape(word_tensor), [embedding_dim]), axis=0)
    )


def get_label_ids(label_tensor, labels_names):
    return tf.cast(tf.reshape(
        tf.feature_column.input_layer(
            {"labels": tf.reshape(label_tensor, [-1])},
            [tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    "labels", labels_names, default_value=0
                ), dimension=1, trainable=False,
                initializer=tf.initializers.constant([list(range(len(labels_names)))])
            )]
        ),
        tf.shape(label_tensor)
    ), tf.int32)


def model_fn(input_values, embedding_words, embedding_matrix, label_vocab):
    (sentence_lines, padded_sentences, padded_sentence_len), \
        (padded_words, padded_word_len, padded_char_bytes), \
        padded_labels = input_values

    char_input = get_char_input(padded_char_bytes, CHAR_EMBEDDING_DIM)
    word_input = get_word_input(padded_sentences, embedding_words, embedding_matrix)
    label_ids = get_label_ids(padded_labels, label_vocab)

    tf_dropout_rate = tf.placeholder_with_default(0.0, shape=[])

    tf_char_vectors = tf.reshape(char_input, tf.concat(([-1], tf.shape(char_input)[2:]), axis=0))
    tf_char_seq_len = tf.reshape(padded_word_len, [-1])
    tf_word_vectors = word_input
    tf_word_seq_len = padded_sentence_len
    tf_max_sentence_len = tf.shape(word_input)[1]
    tf_labels = label_ids

    ((_, _), (tf_char_output_state_fw, tf_char_output_state_bw)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=rnn.BasicLSTMCell(CHAR_LSTM_UNITS),
        cell_bw=rnn.BasicLSTMCell(CHAR_LSTM_UNITS),
        inputs=tf_char_vectors, sequence_length=tf_char_seq_len,
        dtype=tf.float32, scope="char_rnn"
    )

    tf_char_output_states = tf.concat([tf_char_output_state_fw[1], tf_char_output_state_bw[1]], axis=1)
    tf_char_outputs = tf.reshape(tf_char_output_states, (-1, tf_max_sentence_len, CHAR_LSTM_UNITS * 2))
    tf_char_outputs_dropped = tf.layers.dropout(
        tf_char_outputs, rate=tf_dropout_rate, training=tf.greater(tf_dropout_rate, 0)
    )

    tf_word_inputs = tf.concat([tf_word_vectors, tf_char_outputs_dropped], axis=2)

    ((tf_word_outputs_fw, tf_word_outputs_bw), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=rnn.DropoutWrapper(rnn.BasicLSTMCell(WORD_LSTM_UNITS), output_keep_prob=1.0-tf_dropout_rate),
        cell_bw=rnn.DropoutWrapper(rnn.BasicLSTMCell(WORD_LSTM_UNITS), output_keep_prob=1.0-tf_dropout_rate),
        inputs=tf_word_inputs, sequence_length=tf_word_seq_len,
        dtype=tf.float32, scope="word_rnn"
    )

    tf_word_outputs = tf.concat([tf_word_outputs_fw, tf_word_outputs_bw], axis=2)

    tf_logits = tf.layers.dense(tf_word_outputs, len(label_vocab), activation=None)
    tf_log_likelihoods, tf_transitions = crf.crf_log_likelihood(tf_logits, tf_labels, tf_word_seq_len)
    tf_loss = -tf.reduce_mean(tf_log_likelihoods)

    tf_predictions, _ = crf.crf_decode(tf_logits, tf_transitions, tf_word_seq_len)

    tf_real_word_seq_len = tf_word_seq_len - 2
    tf_real_predictions = tf_predictions[:, 1:-1]
    tf_real_labels = tf_labels[:, 1:-1]

    tf_word_mask = tf.sequence_mask(tf_real_word_seq_len)
    tf_masked_predictions = tf.boolean_mask(tf_real_predictions, tf_word_mask)
    tf_masked_labels = tf.boolean_mask(tf_real_labels, tf_word_mask)

    tf_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf_masked_predictions, tf_masked_labels), tf.float32))
    tf_train = tf.train.AdamOptimizer().minimize(tf_loss)

    return tf_train, tf_loss, tf_accuracy, \
        tf_real_predictions, tf_real_labels, tf_real_word_seq_len, \
        sentence_lines, tf_dropout_rate


def extract_entities(data_labels, seq_len, num_labels, num_classes):
    with_classes = set([])
    without_classes = set([])

    for i in range(len(data_labels)):
        current = None
        sentence = data_labels[i][:seq_len[i]]
        for l in range(len(sentence)):
            lbl = sentence[l]
            if current is not None and \
               (lbl < num_classes or lbl == num_labels - 1 or current[2] != lbl - num_classes):
                with_classes.add(current + (l-1,))
                without_classes.add(current[:-1] + (l-1,))
                current = None
            if lbl < num_classes:
                current = (i, l, lbl)
            elif lbl < num_labels - 1 and current is None:
                current = (i, l, lbl-num_classes)
        if current is not None:
            with_classes.add(current + (len(sentence)-1,))
            without_classes.add(current[:-1] + (len(sentence)-1,))

    return with_classes, without_classes


def compute_f1_score(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec * 100, rec * 100, f1 * 100


def fix_labels(label_matrix, seq_len, num_labels):
    result = label_matrix[:]
    for i in range(len(seq_len)):
        for j in range(seq_len[i]):
            result[i, j] = result[i, j] - 1 if result[i, j] > 0 else num_labels - 1
    return result


def compute_diagnostic_data(gold, predicted, seq_len, num_labels):
    num_classes = num_labels // 2

    gold = fix_labels(gold, seq_len, num_labels)
    predicted = fix_labels(predicted, seq_len, num_labels)

    conf = np.zeros([num_labels, num_labels], dtype=np.int32)
    for i in range(len(gold)):
        for j in range(seq_len[i]):
            conf[gold[i, j], predicted[i, j]] += 1

    acc = np.sum(np.diag(conf)) / sum(seq_len) * 100

    b_tp = np.sum(conf[:num_classes, :num_classes])
    b_tn = np.sum(conf[num_classes:, num_classes:])
    b_fp = np.sum(conf[num_classes:, :num_classes])
    b_fn = np.sum(conf[:num_classes, num_classes:])
    b_prec, b_rec, b_f1 = compute_f1_score(b_tp, b_fp, b_fn)

    b_acc = np.sum(np.diag(conf[:num_classes])) / np.sum(conf[:num_classes]) * 100
    e_acc = np.sum(np.diag(conf[:num_labels-1])) / np.sum(conf[:num_labels-1]) * 100
    o_acc = conf[-1, -1] / np.sum(conf[-1]) * 100

    gold_entities, gold_spans = extract_entities(gold, seq_len, num_labels, num_classes)
    predicted_entities, predicted_spans = extract_entities(predicted, seq_len, num_labels, num_classes)

    e_tp = len([1 for p in predicted_spans if p in gold_spans])
    e_fp = len([1 for p in predicted_spans if p not in gold_spans])
    e_fn = len([1 for p in gold_spans if p not in predicted_spans])
    e_prec, e_rec, e_f1 = compute_f1_score(e_tp, e_fp, e_fn)

    ec_tp = len([1 for p in predicted_entities if p in gold_entities])
    ec_fp = len([1 for p in predicted_entities if p not in gold_entities])
    ec_fn = len([1 for p in gold_entities if p not in predicted_entities])
    ec_prec, ec_rec, ec_f1 = compute_f1_score(ec_tp, ec_fp, ec_fn)

    result = {
        "acc": acc, "B_acc": b_acc, "E_acc": e_acc, "O_acc": o_acc,
        "B_TP": b_tp, "B_TN": b_tn, "B_FP": b_fp, "B_FN": b_fn,
        "B_prec": b_prec, "B_rec": b_rec, "B_F1": b_f1,
        "E_TP": e_tp, "E_FP": e_fp, "E_FN": e_fn,
        "E_prec": e_prec, "E_rec": e_rec, "E_F1": e_f1,
        "EC_TP": ec_tp, "EC_FP": ec_fp, "EC_FN": ec_fn,
        "EC_prec": ec_prec, "EC_rec": ec_rec, "EC_F1": ec_f1,
        "confusion": conf
    }

    for i in range(num_classes):
        ec_class_tp = len([1 for p in predicted_entities if p[2] == i and p in gold_entities])
        ec_class_fp = len([1 for p in predicted_entities if p[2] == i and p not in gold_entities])
        ec_class_fn = len([1 for p in gold_entities if p[2] == i and p not in predicted_entities])
        ec_class_prec, ec_class_rec, ec_class_f1 = compute_f1_score(ec_class_tp, ec_class_fp, ec_class_fn)

        result["EC_" + str(i) + "_TP"] = ec_class_tp
        result["EC_" + str(i) + "_FP"] = ec_class_fp
        result["EC_" + str(i) + "_FN"] = ec_class_fn
        result["EC_" + str(i) + "_prec"] = ec_class_prec
        result["EC_" + str(i) + "_rec"] = ec_class_rec
        result["EC_" + str(i) + "_F1"] = ec_class_f1

    return result


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
            print(step)

        val_acc, val_loss, val_labels, val_predictions, val_sentence_len = sess.run(
            [accuracy, loss, labels, predictions, sentence_length],
            feed_dict={
                data_handle: val_handle
            }
        )

        if REPORT_IOB_F1_SCORE:
            diag_data = compute_diagnostic_data(
                val_labels, val_predictions, val_sentence_len, len(label_names)
            )

            print("{:<16} {:<39} {:<30} {:<30} {}".format(
                "{0}.val: L {1:.3f}".format(epoch + 1, val_loss),
                "acc [B {d[B_acc]:.2f} E {d[E_acc]:.2f} O {d[O_acc]:.2f} T {d[acc]:.2f}]".format(d=diag_data),
                "B [P {d[B_prec]:.2f} R {d[B_rec]:.2f} F1 {d[B_F1]:.2f}]".format(d=diag_data),
                "E [P {d[E_prec]:.2f} R {d[E_rec]:.2f} F1 {d[E_F1]:.2f}]".format(d=diag_data),
                "EC [P {d[EC_prec]:.2f} R {d[EC_rec]:.2f} F1 {d[EC_F1]:.2f}]".format(d=diag_data),
            ))
        else:
            echo(epoch + 1, "val acc", val_acc)
