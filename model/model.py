import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.crf as crf
import tensorflow.contrib.lookup as lookup


def get_char_embeddings(char_tensor, embedding_dim):
    """Convert an int tensor of character bytes into a float32 tensor of char (byte) embeddings."""
    char_embeddings = tf.get_variable(
        name="char_embeddings", shape=[256, embedding_dim],
        initializer=tf.initializers.random_normal(),
        trainable=True
    )

    return tf.nn.embedding_lookup(char_embeddings, tf.cast(char_tensor, tf.int32))


def get_word_embeddings(word_tensor, embedding_words_file, embedding_matrix):
    """Convert a string tensor of words into a float32 tensor of word embeddings."""
    word_lookup = lookup.index_table_from_file(embedding_words_file, default_value=0)
    word_ids = word_lookup.lookup(word_tensor)

    word_embeddings = tf.concat((
        tf.get_variable(name="unknown_word_embedding", initializer=embedding_matrix[:1], trainable=True),
        tf.get_variable(name="known_word_embeddings", initializer=embedding_matrix[1:], trainable=False)
    ), axis=0)

    return tf.nn.embedding_lookup(word_embeddings, word_ids)


def get_label_ids(label_tensor, labels_names):
    """Convert a string tensor of string label names into an int32 tensor of label indices (same shape)."""
    label_lookup = lookup.index_table_from_tensor(labels_names, default_value=0)
    label_ids = label_lookup.lookup(label_tensor)

    return tf.cast(label_ids, tf.int32)


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


def model_fn(input_values, embedding_words_file, embedding_matrix, label_vocab, training=True,
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
    word_embeddings = get_word_embeddings(unique_words, embedding_words_file, embedding_matrix)

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

    # removing batch-padded empty words
    # from predictions and labels
    word_mask = tf.sequence_mask(word_seq_len)
    masked_predictions = tf.boolean_mask(predictions, word_mask)
    masked_labels = tf.boolean_mask(labels, word_mask)

    # accuracy (single number) and training op (using Adam)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(masked_predictions, masked_labels), tf.float32))
    train_op = tf.train.AdamOptimizer().minimize(loss) if training else None

    return train_op, loss, accuracy, \
        predictions, labels, word_seq_len, \
        raw_sentences, dropout_rate
