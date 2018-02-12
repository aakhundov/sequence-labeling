import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.crf as crf


def get_char_embeddings(char_tensor, embedding_dim):
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


def get_word_embeddings(word_tensor, embedding_words, embedding_matrix):
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


def create_layered_bi_lstm(num_layers, num_units, dropout_rate):
    layers_fw, layers_bw = [], []
    for _ in range(num_layers):
        for layers in [layers_fw, layers_bw]:
            layers.append(
                rnn.DropoutWrapper(
                    rnn.BasicLSTMCell(num_units),
                    output_keep_prob=1.0-dropout_rate
                )
            )

    if num_layers == 1:
        return layers_fw[0], layers_bw[0]
    else:
        return rnn.MultiRNNCell(layers_fw), rnn.MultiRNNCell(layers_bw)


def model_fn(input_values, embedding_words, embedding_matrix, label_vocab,
             char_lstm_units=64, word_lstm_units=128, char_embedding_dim=50,
             char_lstm_layers=1, word_lstm_layers=1):

    (raw_sentences, sentence_tokens, sentence_len, label_tokens), \
        ((unique_words, unique_word_idx), unique_word_len, unique_word_bytes) = input_values

    tf_dropout_rate = tf.placeholder_with_default(0.0, shape=[])

    tf_char_embeddings = get_char_embeddings(unique_word_bytes, char_embedding_dim)
    tf_word_embeddings = get_word_embeddings(unique_words, embedding_words, embedding_matrix)

    tf_char_inputs = tf_char_embeddings
    tf_char_seq_len = tf.reshape(unique_word_len, [-1])
    tf_char_lstm_fw, tf_char_lstm_bw = create_layered_bi_lstm(
        char_lstm_layers, char_lstm_units, tf_dropout_rate
    )

    ((tf_char_outputs_fw, tf_char_outputs_bw), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf_char_lstm_fw, cell_bw=tf_char_lstm_bw,
        inputs=tf_char_inputs, sequence_length=tf_char_seq_len,
        dtype=tf.float32, scope="char_rnn"
    )

    tf_char_outputs = tf.reshape(
        tf.concat([
            tf.gather_nd(
                tf_char_outputs_fw,
                tf.transpose(tf.stack((
                    tf.range(tf.shape(tf_char_seq_len)[0]),
                    tf.maximum(0, tf_char_seq_len - 1)
                )))
            ),
            tf_char_outputs_bw[:, 0, :]
        ], axis=1),
        [-1, char_lstm_units * 2]
    )

    tf_unique_word_features = tf.concat([tf_word_embeddings, tf_char_outputs], axis=1)
    tf_sentence_word_features = tf.map_fn(lambda i: tf_unique_word_features[i], unique_word_idx,
                                          dtype=tf_unique_word_features.dtype)

    tf_word_inputs = tf.reshape(tf_sentence_word_features, tf.concat(
        (tf.shape(sentence_tokens)[:2], tf.shape(tf_sentence_word_features)[1:]), axis=0
    ))
    tf_word_seq_len = sentence_len
    tf_word_lstm_fw, tf_word_lstm_bw = create_layered_bi_lstm(
        word_lstm_layers, word_lstm_units, tf_dropout_rate
    )

    ((tf_word_outputs_fw, tf_word_outputs_bw), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf_word_lstm_fw, cell_bw=tf_word_lstm_bw,
        inputs=tf_word_inputs, sequence_length=tf_word_seq_len,
        dtype=tf.float32, scope="word_rnn"
    )

    tf_word_outputs = tf.concat([tf_word_outputs_fw, tf_word_outputs_bw], axis=2)

    tf_labels = get_label_ids(label_tokens, label_vocab)
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
        raw_sentences, tf_dropout_rate
