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


def get_word_embeddings(word_tensor, embedding_words, embedding_vectors):
    """Convert a string tensor of words into a float32 tensor of word embeddings."""
    word_lookup = lookup.index_table_from_tensor(embedding_words, default_value=0)
    word_ids = word_lookup.lookup(word_tensor)

    word_embeddings = tf.concat((
        tf.get_variable(name="unknown_word_embedding", initializer=embedding_vectors[:1], trainable=True),
        tf.get_variable(name="known_word_embeddings", initializer=embedding_vectors[1:], trainable=False)
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


def model_fn(input_values, label_vocab, embedding_words, embedding_vectors,
             char_lstm_units, word_lstm_units, char_lstm_layers, word_lstm_layers, char_embedding_dim,
             training=False, initial_learning_rate=0.001, lr_decay_rate=0.05,
             use_char_embeddings=True, use_crf_layer=True):

    # destructuring compound input values into components
    (raw_sentences, sentence_tokens, sentence_len, label_tokens), \
        ((unique_words, unique_word_index), unique_word_len, unique_word_bytes) = input_values

    # placeholders, to be set externally (returned)
    dropout_rate = tf.placeholder_with_default(0.0, shape=[])
    completed_epochs = tf.placeholder_with_default(0.0, shape=[])

    # character (byte) and word embeddings created with the helper methods
    # embeddings are created (and processed) only once for each words in a batch
    char_embeddings = get_char_embeddings(unique_word_bytes, char_embedding_dim)
    word_embeddings = get_word_embeddings(unique_words, embedding_words, embedding_vectors)

    # dropping out char embeddings
    dropped_char_embeddings = tf.layers.dropout(
        char_embeddings, dropout_rate,
        training=tf.greater(dropout_rate, 0.0)
    )

    # dropping out word embeddings
    dropped_word_embeddings = tf.layers.dropout(
        word_embeddings, dropout_rate,
        training=tf.greater(dropout_rate, 0.0)
    )

    # char-bi-LSTM configuration
    char_inputs = dropped_char_embeddings
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

    if use_char_embeddings:
        # combining the (computed) char and word embedding features for unique words in a batch
        unique_word_features = tf.concat([dropped_word_embeddings, last_char_outputs], axis=1)
    else:
        # using only word embeddings
        unique_word_features = dropped_word_embeddings

    # expanding unique words into the sentence structure of sentence tensor
    # (unique_word_index pointing from unique words to sentence positions)
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

    # deriving logits with a linear layer
    # applied to word-bi-LSTM outputs
    labels = get_label_ids(label_tokens, label_vocab)
    logits = tf.layers.dense(word_outputs, len(label_vocab), activation=None)

    word_mask = tf.sequence_mask(word_seq_len)

    if use_crf_layer:
        # inference by applying a CRF (and Viterbi decode)
        log_likelihoods, transitions = crf.crf_log_likelihood(logits, labels, word_seq_len)
        predictions, _ = crf.crf_decode(logits, transitions, word_seq_len)
        # minimizing negative log-likelihood of CRF predictions
        loss = -tf.reduce_mean(log_likelihoods)
    else:
        # inference directly from logits by taking argmax
        predictions = tf.cast(tf.argmax(logits, axis=2), tf.int32)
        # minimizing softmax cross-entropy loss masked by word sequence lengths
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(tf.boolean_mask(losses, word_mask))

    # removing batch-padded empty words from predictions and labels
    masked_predictions = tf.boolean_mask(predictions, word_mask)
    masked_labels = tf.boolean_mask(labels, word_mask)

    # accuracy (single number) and training op (using Adam)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(masked_predictions, masked_labels), tf.float32))

    # learning rate with per-epoch LR / (1 + epochs * rate) decay schedule
    learning_rate = initial_learning_rate / (1.0 + lr_decay_rate * completed_epochs)

    # training op through minimizing given loss (CRF or X-entropy) with Adam optimizer
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss) if training else None

    return train_op, loss, accuracy, \
        predictions, labels, word_seq_len, \
        raw_sentences, dropout_rate, completed_epochs
