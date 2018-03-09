import tensorflow as tf


def input_fn(input_lines, batch_size=None, lower_case_words=False,
             shuffle=False, cache=True, repeat=True, num_threads=4):
    """Convert 1D string tf.data.Dataset input_lines into an input pipeline."""

    def split_string(s, delimiter, skip_empty=True):
        """Split a single string tensor s into multiple string tokens by delimiter."""
        return tf.string_split([s], delimiter, skip_empty=skip_empty).values

    def decode_word(word, max_len):
        """Convert string tensor word into a list of encoding bytes zero-padded up to max_len."""
        w_bytes = tf.concat(([1], tf.decode_raw(word, tf.uint8), [2]), axis=0)
        padded = tf.pad(w_bytes, [[0, max_len - tf.shape(w_bytes)[0]]])
        return padded

    def get_word_lengths(words, padding=2):
        """Compute a length of each word in a 1D string tensor."""
        return tf.map_fn(
            lambda x: tf.size(tf.string_split([x], "")) + padding,
            words, dtype=tf.int32
        )

    def get_word_bytes(words, max_len):
        """Retrieve UTF-8 bytes of each word in a 1D string tensor."""
        return tf.map_fn(
            lambda x: decode_word(x, max_len),
            words, dtype=tf.uint8
        )

    def words_to_lower(words):
        """Convert all strings in a 1D tensor to lower-case (same shape tensor is returned)."""
        def to_lower(byte_string):
            return str(byte_string, encoding="utf-8").lower()

        return tf.reshape(tf.map_fn(
            lambda x: tf.py_func(
                to_lower, [x], tf.string, stateful=False
            ), words
        ), [-1])

    # splitting input lines into sentence and label parts (by "\t")
    # extra "\t" is added to create labels placeholder if line contains no labels
    data = input_lines.map(lambda l: split_string(l + "\t", "\t", False), num_threads)
    # splitting sentence and label parts into respective tokens (by " ")
    data = data.map(lambda sp: (sp[0], split_string(sp[0], " "), split_string(sp[1], " ")), num_threads)
    # adding sentence lengths; result: (full sentences, sentence tokens, sentence length, label tokens)
    data = data.map(lambda sl, st, lt: (sl, st, tf.shape(st)[0], lt), num_threads)

    if shuffle:
        if cache:
            # if caching is required, it is
            # done before shuffling to maintain
            # different batches in every epoch
            data = data.cache()
        # shuffling the entire dataset
        data = data.shuffle(1000000000)

    # generating padded batch_size-batches of everything so far
    # or a single batch of the entire dataset if batch_size=None
    data = data.padded_batch(
        batch_size if batch_size is not None else 1000000000,
        ([], [-1], [], [-1]), ("", "", 0, "")
    )

    # adding a tuple of unique words in a batch and their respective indices
    data = data.map(lambda *d: (d, tf.unique(tf.reshape(d[1], [-1]))), num_threads)
    # reshaping unique words' index (resulting from tf.unique) to 2D sentence tokens' shape
    data = data.map(lambda d, u: (d, (u[0], tf.reshape(u[1], tf.shape(d[1])))), num_threads)
    # adding length of each unique word in a batch
    data = data.map(lambda d, u: (d, u, get_word_lengths(u[0])), num_threads)
    # (temporarily) adding the maximum length among unique words
    data = data.map(lambda d, u, uwl: (d, u, uwl, tf.reduce_max(uwl)), num_threads)
    # replacing the maximum length by the 2D tf.uint8 tensor of encoding bytes of unique words
    data = data.map(lambda d, u, uwl, mwl: (d, (u, uwl, get_word_bytes(u[0], mwl))), num_threads)

    if lower_case_words:
        # if required, all unique words are converted to lower case (using Python function)
        data = data.map(lambda d, w: (d, ((words_to_lower(w[0][0]), w[0][1]), w[1], w[2])))

    if not shuffle and cache:
        # if shuffling is not required, caching the
        # final dataset at once (before repeating)
        data = data.cache()

    if repeat:
        # if repeating is required,
        # doing so infinitely
        data = data.repeat()

    return data.prefetch(1)
