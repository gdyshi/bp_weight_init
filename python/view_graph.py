# 本文件出自：
# https://github.com/feixia586/zhihu_material/blob/master/weight_initialization/w_init.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
        # 固定值（0值）
        # W = tf.Variable(tf.zeros([node_in, node_out]))
        # 随机化
        # W = tf.Variable(np.random.randn(node_in, node_out).astype('float32')) * 1
        # 小随机数
        W = tf.Variable(np.random.randn(node_in, node_out).astype('float32')) * 0.01
        # Xavier
        # W = tf.Variable(np.random.randn(node_in, node_out).astype('float32')) / (np.sqrt(node_in))
        # HE/MSRA
        # W = tf.Variable(np.random.randn(node_in, node_out).astype('float32')) / (np.sqrt(node_in/2))
        fc = tf.matmul(X, W)
        # Batch Normalization
        # fc = tf.contrib.layers.batch_norm(fc, center=True, scale=True,
        #                                   is_training=True)
        # tanh激活
        fc = tf.nn.tanh(fc)
        # sigmoid激活
        # fc = tf.nn.sigmoid(fc)
        # relu激活
        # fc = tf.nn.relu(fc)
        fcs.append(fc)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    print('input mean {0:.5f} and std {1:.5f}'.format(np.mean(data.eval()),
                                                      np.std(data.eval())))
    for idx, fc in enumerate(fcs):
        print('layer {0} mean {1:.5f} and std {2:.5f}'.format(idx + 1, np.mean(fc.eval()),
                                                              np.std(fc.eval())))

    plt.figure()
    for idx, fc in enumerate(fcs):
        plt.subplot(1, len(fcs), idx + 1)
        plt.hist(fc.eval().flatten(), 30, range=[-1, 1])
        plt.xlabel('layer ' + str(idx + 1))
        plt.yticks([])
    plt.show()