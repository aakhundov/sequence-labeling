import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.crf as crf


def get_char_embeddings(char_tensor, embedding_dim):
    """Convert an int tensor of character bytes into 2D tensor of char (byte) embeddings."""
    return tf.reshape(
        tf.feature_column.input_layer(
            {"chars": tf.reshape(char_tensor, [-1])},
            [tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity("chars", 256),
                dimension=embedding_dim, trainable=True
            )]
        ),
        tf.concat((tf.shape(char_tensor), [embedding_dim]), axis=0)
    )


def get_word_embeddings(word_tensor, embedding_words, embedding_matrix):
    """Convert a string tensor of words into 2D tensor of word embeddings."""
    _, embedding_dim = embedding_matrix.shape

    return tf.reshape(
        tf.feature_column.input_layer(
            {"sentences": tf.reshape(word_tensor, [-1])},
            [tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    "sentences", embedding_words, default_value=0    # missing words assigned zero index
                ), dimension=embedding_dim, trainable=False,
                initializer=tf.initializers.constant(embedding_matrix)
            )]
        ),
        tf.concat((tf.shape(word_tensor), [embedding_dim]), axis=0)
    )


def get_label_ids(label_tensor, labels_names):
    """Convert a string tensor of human-readable label names into integer tensor of label indices (same shape)."""
    return tf.cast(tf.reshape(
        tf.feature_column.input_layer(
            {"labels": tf.reshape(label_tensor, [-1])},
            [tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    "labels", labels_names, default_value=len(labels_names)-1  # dummy (-) label assigned last index
                ), dimension=1, trainable=False,
                initializer=tf.initializers.constant([list(range(len(labels_names)))])
            )]
        ),
        tf.shape(label_tensor)
    ), tf.int32)


def create_layered_bi_lstm(num_layers, num_units, dropout_rate):
    """Create a pair of LSTM cells (fw and bw) with arbitrary # of layers and units."""
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
        # if 1 layer, return cells themselves
        return layers_fw[0], layers_bw[0]
    else:
        # if multiple layers, wrap cell lists into MultiRNNCell's
        return rnn.MultiRNNCell(layers_fw), rnn.MultiRNNCell(layers_bw)


def model_fn(input_values, embedding_words, embedding_matrix, label_vocab,
             char_lstm_units=64, word_lstm_units=128, char_embedding_dim=50,
             char_lstm_layers=1, word_lstm_layers=1):

    # destructuring compound input values into components
    (raw_sentences, sentence_tokens, sentence_len, label_tokens), \
        ((unique_words, unique_word_index), unique_word_len, unique_word_bytes) = input_values

    # placeholder for dropout rate, to be set externally (returned)
    dropout_rate = tf.placeholder_with_default(0.0, shape=[])

    # character (byte) and word embeddings created with the helper methods
    # embeddings are created (and processed) only once for each words in a batch
    char_embeddings = get_char_embeddings(unique_word_bytes, char_embedding_dim)
    word_embeddings = get_word_embeddings(unique_words, embedding_words, embedding_matrix)

    # char-bi-LSTM configuration
    char_inputs = char_embeddings
    char_seq_len = tf.reshape(unique_word_len, [-1])
    char_lstm_fw, char_lstm_bw = create_layered_bi_lstm(
        char_lstm_layers, char_lstm_units, dropout_rate
    )

    # char-bi-LSTM: input - bytes (time-steps) of each word (batch item)
    ((char_outputs_fw, char_outputs_bw), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=char_lstm_fw, cell_bw=char_lstm_bw,
        inputs=char_inputs, sequence_length=char_seq_len,
        dtype=tf.float32, scope="char_rnn"
    )

    # fetching a pair of (fw and bw) last outputs for every
    # item in the batch (used as char-embedding of a word)
    last_char_outputs = tf.reshape(
        tf.concat([
            tf.gather_nd(
                char_outputs_fw,
                tf.transpose(tf.stack((
                    tf.range(tf.shape(char_seq_len)[0]),
                    tf.maximum(0, char_seq_len - 1)
                )))
            ),
            char_outputs_bw[:, 0, :]
        ], axis=1),
        [-1, char_lstm_units * 2]
    )

    # combining the features computed for unique word in a batch
    # and expanding them into the sentence structure of sentence tensor
    # (unique_word_index pointing from unique words to sentence positions)
    unique_word_features = tf.concat([word_embeddings, last_char_outputs], axis=1)
    sentence_word_features = tf.gather(unique_word_features, unique_word_index)

    # word-bi-LSTM configuration
    word_inputs = sentence_word_features
    word_seq_len = sentence_len
    word_lstm_fw, word_lstm_bw = create_layered_bi_lstm(
        word_lstm_layers, word_lstm_units, dropout_rate
    )

    # word-bi-LSTM: input - words (time-steps) of each sentences (batch item)
    ((word_outputs_fw, word_outputs_bw), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=word_lstm_fw, cell_bw=word_lstm_bw,
        inputs=word_inputs, sequence_length=word_seq_len,
        dtype=tf.float32, scope="word_rnn"
    )

    # outputs from each time step are used for downstream inference
    word_outputs = tf.concat([word_outputs_fw, word_outputs_bw], axis=2)

    # logits -> CRF -> likelihoods -> loss
    labels = get_label_ids(label_tokens, label_vocab)
    logits = tf.layers.dense(word_outputs, len(label_vocab), activation=None)
    log_likelihoods, transitions = crf.crf_log_likelihood(logits, labels, word_seq_len)
    loss = -tf.reduce_mean(log_likelihoods)

    # predictions decoded from logits and CRF transitions (using Viterbi)
    predictions, _ = crf.crf_decode(logits, transitions, word_seq_len)

    # removing start- and end-of-sentence tokens
    # from sentence lengths, predictions, and labels
    word_seq_len_wo_padding = word_seq_len - 2
    predictions_wo_padding = predictions[:, 1:-1]
    labels_wo_padding = labels[:, 1:-1]

    # removing batch-padded empty words
    # from predictions and labels
    word_mask = tf.sequence_mask(word_seq_len_wo_padding)
    masked_predictions = tf.boolean_mask(predictions_wo_padding, word_mask)
    masked_labels = tf.boolean_mask(labels_wo_padding, word_mask)

    # accuracy (single number) and training op (using Adam)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(masked_predictions, masked_labels), tf.float32))
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return train_op, loss, accuracy, \
        predictions_wo_padding, labels_wo_padding, word_seq_len_wo_padding, \
        raw_sentences, dropout_rate
