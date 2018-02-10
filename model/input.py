import tensorflow as tf


def input_fn(input_lines, batch_size=None, shuffle_data=False):
    sentence_lines = input_lines.map(lambda x: tf.string_split([x], "\t").values[0])
    sentences_tokens = sentence_lines.map(lambda x: tf.string_split([x], " ").values)
    padded_sentences = sentences_tokens.map(lambda x: tf.concat((["<S>"], x, ["</S>"]), axis=0))
    padded_sentence_length = padded_sentences.map(lambda x: tf.shape(x)[0])

    label_lines = input_lines.map(lambda x: tf.string_split([x], "\t").values[-1])
    label_tokens = label_lines.map(lambda x: tf.string_split([x], " ").values)
    padded_labels = label_tokens.map(lambda x: tf.concat(([""], x, [""]), axis=0))

    words = padded_sentences.map(lambda x: tf.reshape(x, [-1, 1]))
    padded_word_length = words.map(
        lambda x: tf.map_fn(lambda y: tf.size(tf.string_split(y, "")) + 2, x, dtype=tf.int32), 4
    )

    def decode_word(word, max_len):
        w_bytes = tf.concat(([1], tf.decode_raw(word[0], tf.uint8)[:max_len], [2]), axis=0)
        padded = tf.pad(w_bytes, [[0, max_len - tf.shape(w_bytes)[0]]])
        return padded

    max_padded_word_length = padded_word_length.map(lambda x: tf.reduce_max(x))
    padded_word_bytes = tf.data.Dataset.zip((words, max_padded_word_length)).map(
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
            data = data.shuffle(1000000000)
        return data.apply(make_padded_batches).repeat().prefetch(1)
    else:
        return data.apply(make_padded_batches).cache().repeat()
