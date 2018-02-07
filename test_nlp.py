import pickle
import tensorflow as tf


BATCH_SIZE = 32
CHAR_EMBEDDING_DIM = 50

EPOCHS = 3
STEPS_PER_EPOCH = 2000


def read_polyglot_data(file_name):
    return pickle.load(open("polyglot/{}.pkl".format(file_name), "rb"), encoding="bytes")


def input_pipeline(input_file, batch_size=1000000000, shuffle_data=False):
    lines = tf.data.TextLineDataset(input_file)

    sentence_lines = lines.map(lambda x: tf.string_split([x], "\t").values[0], 4)
    sentences = sentence_lines.map(lambda x: tf.string_split([x], " ").values, 4)
    bounded_sentences = sentences.map(lambda x: tf.concat((["<S>"], x, ["</S>"]), axis=0), 4)
    sentence_length = bounded_sentences.map(lambda x: tf.shape(x)[0])

    label_lines = lines.map(lambda x: tf.string_split([x], "\t").values[-1], 4)
    labels = label_lines.map(lambda x: tf.concat(([""], tf.string_split([x], " ").values, [""]), axis=0), 4)

    words = bounded_sentences.map(lambda x: tf.reshape(x, [-1, 1]), 4)
    word_byte_length = words.map(
        lambda x: tf.map_fn(lambda y: tf.size(tf.string_split(y, "")) + 2, x, dtype=tf.int32), 4
    )

    def decode_word(word, max_len):
        w_bytes = tf.concat(([1], tf.decode_raw(word[0], tf.uint8)[:max_len], [2]), axis=0)
        padded = tf.pad(w_bytes, [[0, max_len - tf.shape(w_bytes)[0]]])
        return padded

    max_byte_length = word_byte_length.map(lambda x: tf.reduce_max(x), 4)
    word_bytes = tf.data.Dataset.zip((words, max_byte_length)).map(
        lambda x, y: tf.map_fn(lambda z: decode_word(z, y), x, dtype=tf.uint8), 4
    )

    data = tf.data.Dataset.zip((
        tf.data.Dataset.zip((sentence_lines, bounded_sentences, sentence_length)),
        tf.data.Dataset.zip((words, word_byte_length, word_bytes)),
        labels
    )).cache()

    if shuffle_data:
        data = data.shuffle(batch_size * 128)

    return data.padded_batch(
        batch_size,
        (([], [-1], []), ([-1, -1], [-1], [-1, -1]), [-1]),
        (("", "", 0), ("", 0, tf.constant(0, tf.uint8)), "")
    ).repeat()


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


def get_word_input(word_tensor, word_list, embedding_matrix):
    _, embedding_dim = embedding_matrix.shape

    return tf.reshape(
        tf.feature_column.input_layer(
            {"sentences": tf.reshape(word_tensor, [-1])},
            [tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    "sentences", word_list, default_value=0
                ), dimension=embedding_dim, trainable=False,
                # initializer=tf.initializers.constant(embedding_matrix)
            )]
        ),
        tf.concat((tf.shape(word_tensor), [embedding_dim]), axis=0)
    )


def get_label_ids(label_tensor, labels):
    return tf.cast(tf.reshape(
        tf.feature_column.input_layer(
            {"labels": tf.reshape(label_tensor, [-1])},
            [tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    "labels", labels, default_value=0
                ), dimension=1, trainable=False,
                initializer=tf.initializers.constant([list(range(len(labels)))])
            )]
        ),
        tf.shape(label_tensor)
    ), tf.int32)


print("A")

with tf.device("/cpu:0"):
    train_data = input_pipeline("data/train.txt", batch_size=BATCH_SIZE, shuffle_data=True)
    val_data = input_pipeline("data/val.txt", shuffle_data=False)
    test_data = input_pipeline("data/test.txt", shuffle_data=False)

data_handle = tf.placeholder(tf.string, shape=())
(ln, s, sl), (w, wl, c), lb = tf.data.Iterator.from_string_handle(
    data_handle, train_data.output_types, train_data.output_shapes
).get_next()

print("B")

char_input = get_char_input(c, 50)
word_input = get_word_input(s, *read_polyglot_data("polyglot-en"))
label_names = [line[:-1] for line in open("data/labels.txt").readlines()]
label_ids = get_label_ids(lb, label_names)

print("D")


with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())

    train_handle = sess.run(train_data.make_one_shot_iterator().string_handle())
    val_handle = sess.run(val_data.make_one_shot_iterator().string_handle())
    test_handle = sess.run(test_data.make_one_shot_iterator().string_handle())

    print("E")

    for epoch in range(EPOCHS):
        current = 0
        for step in range(STEPS_PER_EPOCH):
            tln, ts, tsl, tw, twl, tc, tlb, tci, twi, tli = sess.run(
                [ln, s, sl, w, wl, c, lb, char_input, word_input, label_ids],
                feed_dict={data_handle: train_handle}
            )

            """
            print(tln)
            print(ts)
            print(tsl)
            print()

            print(tw)
            print(twl)
            print(tc)
            print()

            print(tlb)
            print(tli)
            print()

            print(tci.shape)
            print(twi.shape)
            print()
            """

            current += 1

            if current % 100 == 0:
                print(current)

        print()
        print("train", current)
        wi, ci, tsl, twl = sess.run([word_input, char_input, sl, wl], feed_dict={data_handle: val_handle})
        print("val", wi.shape, ci.shape, tsl.shape, twl.shape)
        wi, ci, tsl, twl = sess.run([word_input, char_input, sl, wl], feed_dict={data_handle: test_handle})
        print("test", wi.shape, ci.shape, tsl.shape, twl.shape)
        print()

print("F")
