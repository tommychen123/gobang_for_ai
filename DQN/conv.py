#########################

#    卷积网络神经结构    #

#########################

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def conv2d(x, w):
    """定义卷积函数"""
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_3_3(x):
    """定义2*2最大池化层"""
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


def conv_net():
    xs = tf.placeholder(tf.float32, shape=(None, 15*15), name="xs")
    ys = tf.placeholder(tf.float32, shape=(None, 1), name="ys")
    y0 = tf.reshape(xs, [-1, 15, 15, 1])

    # 第一层 卷积
    w1 = tf.Variable(tf.truncated_normal(
        shape=[3, 3, 1, 8], stddev=1),name="w1")
    b1 = tf.Variable(tf.constant(0.1, shape=[8]),name="b1")
    h1 = tf.nn.relu(conv2d(y0,  w1) + b1)
    y1 = max_pool_3_3(h1)  # 5*5*8

    # 第二层 卷积
    w2 = tf.Variable(tf.truncated_normal(
        shape=[3, 3, 8, 16], stddev=1),name="w2")
    b2 = tf.Variable(tf.constant(0.1, shape=[1, 16]),name="b2")
    y2 = tf.nn.relu(conv2d(y1,  w2) + b2)  # 5*5*16

    # 第三层 输出层
    w3 = tf.Variable(tf.truncated_normal(
        shape=[5*5*16, 1], stddev=1),name="w3")
    b3 = tf.Variable(tf.constant(0.1, shape=[1, 1]),name="b3")
    h3 = tf.reshape(y2, [-1, 5 * 5 * 16])

    y = tf.add(tf.matmul(h3, w3), b3, name="y")
    return xs, ys, y
