import argparse
# Copyright 2017 gdyshi. All Rights Reserved.
# github: https://github.com/gdyshi
# ==============================================================================

import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

FLAGS = None
INIT_METHORD_ZERO = "zero"  # 固定值（0值）
INIT_METHORD_RANDOM = "random"  # 随机数
INIT_METHORD_LITTLE_RANDOM = "little_random"  # 小随机数
INIT_METHORD_XAVIER = "Xavier"  # Xavier
INIT_METHORD_HE = "he"  # HE/MSRA

ACTIVATION_FUNCTION_TANH = "tanh"
ACTIVATION_FUNCTION_SIGMOID = "sigmoid"
ACTIVATION_FUNCTION_RELU = "relu"

# layers = [784, 577, 384, 256, 171, 114, 76, 51, 34, 23, 15, 10] # batch norm
# layers = [784, 320, 160, 80, 40, 20, 10]
layers = [784, 390, 156, 62, 25, 10]
# layers = [784, 270, 90, 30, 10]
# layers = [784, 90, 10]

# 初始化方法选择
INIT_METHORD = INIT_METHORD_LITTLE_RANDOM

# 批量归一化
BATCH_NORM = True

# 激活函数选择
ACTIVATION_FUNCTION = ACTIVATION_FUNCTION_SIGMOID


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print(layers)
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    for i in range(0, len(layers) - 1):
        X = x if i == 0 else y
        node_in = layers[i]
        node_out = layers[i + 1]
        if INIT_METHORD == INIT_METHORD_ZERO:
            W = tf.Variable(tf.zeros([node_in, node_out]))
        elif INIT_METHORD == INIT_METHORD_RANDOM:
            W = tf.Variable(np.random.randn(node_in, node_out).astype('float32'))
        elif INIT_METHORD == INIT_METHORD_LITTLE_RANDOM:
            W = tf.Variable(np.random.randn(node_in, node_out).astype('float32') * 0.01)
        elif INIT_METHORD == INIT_METHORD_XAVIER:
            W = tf.Variable(np.random.randn(node_in, node_out).astype('float32') / (np.sqrt(node_in)))
        elif INIT_METHORD == INIT_METHORD_HE:
            W = tf.Variable(np.random.randn(node_in, node_out).astype('float32') / (np.sqrt(node_in / 2)))
        b = tf.Variable(np.random.randn(node_out).astype('float32'))
        z = tf.matmul(X, W) + b

        if BATCH_NORM:
            z = tf.contrib.layers.batch_norm(z, center=True, scale=True,
                                              is_training=True)

        if ACTIVATION_FUNCTION == ACTIVATION_FUNCTION_TANH:
            y = tf.nn.tanh(z)
        elif ACTIVATION_FUNCTION == ACTIVATION_FUNCTION_SIGMOID:
            y = tf.nn.sigmoid(z)
        elif ACTIVATION_FUNCTION == ACTIVATION_FUNCTION_RELU:
            y = tf.nn.relu(z)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    print(y)
    loss = tf.reduce_mean(tf.norm(y_ - y, axis=1) ** 2) / 2
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=z_3))
    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss) #relu 0.01 tanh 0.1
    train_step = tf.train.GradientDescentOptimizer(3.0).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            if i % 1000 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images,
                                                          y_: mnist.test.labels})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='E:\data\mnist',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
