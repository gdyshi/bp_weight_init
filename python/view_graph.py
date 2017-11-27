# 本文件源自：
# https://github.com/feixia586/zhihu_material/blob/master/weight_initialization/w_init.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

INIT_METHORD_ZERO = "zero"  # 固定值（0值）
INIT_METHORD_RANDOM = "random"  # 随机数
INIT_METHORD_LITTLE_RANDOM = "little_random"  # 小随机数
INIT_METHORD_XAVIER = "Xavier"  # Xavier
INIT_METHORD_HE = "he"  # HE/MSRA

ACTIVATION_FUNCTION_TANH = "tanh"
ACTIVATION_FUNCTION_SIGMOID = "sigmoid"
ACTIVATION_FUNCTION_RELU = "relu"

# 初始化方法选择
INIT_METHORD = INIT_METHORD_RANDOM

# 批量归一化
BATCH_NORM = True

# 激活函数选择
ACTIVATION_FUNCTION = ACTIVATION_FUNCTION_TANH

graph = tf.Graph()
with graph.as_default():
    data = tf.constant(np.random.randn(2000, 800).astype('float32'))
    layer_sizes = [800 - 50 * i for i in range(0, 10)]
    num_layers = len(layer_sizes)

    fcs = []
    for i in range(0, num_layers - 1):
        X = data if i == 0 else fcs[i - 1]
        node_in = layer_sizes[i]
        node_out = layer_sizes[i + 1]
        if INIT_METHORD == INIT_METHORD_ZERO:
            W = tf.Variable(tf.zeros([node_in, node_out]))
        elif INIT_METHORD == INIT_METHORD_RANDOM:
            W = tf.Variable(np.random.randn(node_in, node_out).astype('float32'))
        elif INIT_METHORD == INIT_METHORD_LITTLE_RANDOM:
            W = tf.Variable(np.random.randn(node_in, node_out).astype('float32')) * 0.01
        elif INIT_METHORD == INIT_METHORD_XAVIER:
            W = tf.Variable(np.random.randn(node_in, node_out).astype('float32')) / (np.sqrt(node_in))
        elif INIT_METHORD == INIT_METHORD_HE:
            W = tf.Variable(np.random.randn(node_in, node_out).astype('float32')) / (np.sqrt(node_in / 2))

        fc = tf.matmul(X, W)

        if BATCH_NORM:
            fc = tf.contrib.layers.batch_norm(fc, center=True, scale=True,
                                              is_training=True)

        if ACTIVATION_FUNCTION == ACTIVATION_FUNCTION_TANH:
            fc = tf.nn.tanh(fc)
        elif ACTIVATION_FUNCTION == ACTIVATION_FUNCTION_SIGMOID:
            fc = tf.nn.sigmoid(fc)
        elif ACTIVATION_FUNCTION == ACTIVATION_FUNCTION_RELU:
            fc = tf.nn.relu(fc)

        fcs.append(fc)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    print('input mean {0:.5f} and std {1:.5f}'.format(np.mean(data.eval()),
                                                      np.std(data.eval())))
    for idx, fc in enumerate(fcs):
        print('layer {0} mean {1:.5f} and std {2:.5f}'.format(idx + 1, np.mean(fc.eval()),
                                                              np.std(fc.eval())))

    figure = plt.figure()
    figure.suptitle('init methord:' + INIT_METHORD + ',batch norm:' + str(BATCH_NORM) + ',activation function:' + ACTIVATION_FUNCTION)
    for idx, fc in enumerate(fcs):
        plt.subplot(1, len(fcs), idx + 1)
        plt.hist(fc.eval().flatten(), 30, range=[-1, 1])
        plt.xlabel('layer ' + str(idx + 1))
        plt.yticks([])
    plt.show()
