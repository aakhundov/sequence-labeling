import math
import tensorflow as tf


x, y = tf.keras.datasets.mnist.load_data()


EPOCHS = 50
BATCH_SIZE = 32
TRAIN_SIZE = len(x[0])
TEST_SIZE = len(y[0])


train_data = tf.data.Dataset.from_tensor_slices(x).shuffle(TRAIN_SIZE).batch(BATCH_SIZE).repeat(EPOCHS)
whole_train_data = tf.data.Dataset.from_tensor_slices(x).batch(TRAIN_SIZE).repeat(EPOCHS)
test_data = tf.data.Dataset.from_tensor_slices(y).batch(TEST_SIZE).repeat(EPOCHS)

handle = tf.placeholder(tf.string, shape=())
iterator = tf.data.Iterator.from_string_handle(
    handle, train_data.output_types, train_data.output_shapes
)

images, labels = iterator.get_next()
images = tf.cast(images, tf.float32) / 255.
labels = tf.cast(labels, tf.int64)

is_training = tf.placeholder_with_default(False, shape=())
dropout_rate = tf.placeholder_with_default(0.0, shape=())

inputs = tf.cast(images, tf.float32) / 255.
expanded = tf.reshape(inputs, [-1, 28, 28, 1])
flattened = tf.reshape(inputs, [-1, 784])

conv1 = tf.layers.conv2d(expanded, 32, 5)
pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding="same")

conv2 = tf.layers.conv2d(pool1, 64, 5)
pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding="same")
conv_pool2_flat = tf.layers.flatten(pool2)

layer1 = tf.layers.dense(
    conv_pool2_flat, 1024,
    kernel_initializer=tf.initializers.random_normal(stddev=0.1),
    bias_initializer=tf.initializers.random_normal(stddev=0.1),
    activation=tf.nn.relu
)
drop1 = tf.layers.dropout(layer1, rate=dropout_rate, training=is_training)

logits = tf.layers.dense(
    drop1, 10,
    kernel_initializer=tf.initializers.random_normal(stddev=0.1),
    bias_initializer=tf.initializers.random_normal(stddev=0.1)
)

prediction = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
train = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:
    train_handle = sess.run(train_data.make_one_shot_iterator().string_handle())
    whole_train_handle = sess.run(whole_train_data.make_one_shot_iterator().string_handle())
    test_handle = sess.run(test_data.make_one_shot_iterator().string_handle())

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for e in range(EPOCHS):
        for i in range(int(math.ceil(TRAIN_SIZE / BATCH_SIZE))):
            sess.run(train, feed_dict={handle: train_handle, is_training: True, dropout_rate: 0.5})

        print("epoch {}, val loss {:.4f}, val acc {:.4f}, train loss {:.4f}, train acc {:.4f}".format(
            e+1, *sess.run([loss, accuracy], feed_dict={handle: test_handle}),
            *sess.run([loss, accuracy], feed_dict={handle: whole_train_handle})
        ))
