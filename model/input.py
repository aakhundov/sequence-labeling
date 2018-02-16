import tensorflow as tf


def input_fn(input_lines, batch_size=None, shuffle=False, cache=True, repeat=True, num_threads=4):
    """Process 1D string tensor input_lines into an input pipeline."""

    def split_string(s, delimiter):
        """Split a single string tensor s into multiple string tokens by delimiter."""
        return tf.string_split([s], delimiter).values

    def pad_tokens(pre, tokens, post):
        """Add pre-token and pos-token to a list of tokens."""
        return tf.concat(([pre], tokens, [post]), axis=0)

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
        """Retrieve encoding bytes of each word in a 1D string tensor."""
        return tf.map_fn(
            lambda x: decode_word(x, max_len),
            words, dtype=tf.uint8
        )

    # splitting input lines into sentence and label parts (by "\t")
    data = input_lines.map(lambda l: split_string(l, "\t"), num_threads)
    # splitting sentence and label parts into respective tokens (by " ")
    data = data.map(lambda sp: (sp[0], split_string(sp[0], " "), split_string(sp[1], " ")), num_threads)
    # padding sentences (with begin- and end-of-sentence tokens) and labels respectively (with empty labels)
    data = data.map(lambda sl, st, lt: (sl, pad_tokens("<S>", st, "</S>"), pad_tokens("-", lt, "-")), num_threads)
    # adding sentence lengths; result: (full sentences, sentence tokens, sentence length, label tokens)
    data = data.map(lambda sl, pst, plt: (sl, pst, tf.shape(pst)[0], plt), num_threads)

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

    if not shuffle and cache:
        # if shuffling is not required, caching the
        # final dataset at once (before repeating)
        data = data.cache()

    if repeat:
        # if repeating is required,
        # doing so infinitely
        data = data.repeat()

    return data.prefetch(1)
