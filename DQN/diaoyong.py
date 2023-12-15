import tensorflow.compat.v1 as tf
import numpy as np
import os
tf.disable_v2_behavior()
saver = tf.train.Saver()
with tf.Session() as sess:
    # new_saver = tf.train.import_meta_graph(dir+"/checkPoint/model.meta")
    saver.restore(sess, 'mnist_model.ckpt')
    print("Model restored.")

