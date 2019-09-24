import argparse

import tensorflow as tf
import numpy as np


class MNISTSmall(object):
    def __init__(self):
        pass

    def _encoder(self, x_input, keep_prob, is_train):
        # if not self.varInit:
        #     self.variableInit()
        self.w1 = tf.get_variable("w1", initializer=tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1))
        self.b1 = tf.get_variable("b1", [32], initializer=tf.zeros_initializer)

        self.w2 = tf.get_variable("w2", initializer=tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1))
        self.b2 = tf.get_variable("b2", [64], initializer=tf.zeros_initializer)

        self.w3 = tf.get_variable("w3", initializer=tf.truncated_normal(shape=[7 * 7 * 64, 1024], stddev=0.1))
        self.b3 = tf.get_variable("b3", [1024], initializer=tf.zeros_initializer)

        self.w4 = tf.get_variable("w4", initializer=tf.truncated_normal(shape=[1024, 10], stddev=0.1))
        self.b4 = tf.get_variable("b4", [10], initializer=tf.zeros_initializer)


        conv1 = tf.nn.conv2d(x_input, self.w1, [1, 1, 1, 1], "SAME", name="conv1")
        conv1_bias = tf.nn.bias_add(conv1, self.b1, name="conv1_bias")
        conv1_relu = tf.nn.relu(conv1_bias, name="conv1_relu")

        pool1 = tf.nn.max_pool(conv1_relu, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name="pool1")

        conv2 = tf.nn.conv2d(pool1, self.w2, [1, 1, 1, 1], "SAME", name="conv2")
        conv2_bias = tf.nn.bias_add(conv2, self.b2, name="conv2_bias")
        conv2_relu = tf.nn.relu(conv2_bias, name="conv2_relu")

        pool2 = tf.nn.max_pool(conv2_relu, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name="pool2")

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        fc1 = tf.matmul(pool2_flat, self.w3, name="fc1")
        fc1_bias = tf.nn.bias_add(fc1, self.b3, name="fc1_bias")
        fc1_relu = tf.nn.relu(fc1_bias, name="fc1_relu")

        drop_fc1 = tf.nn.dropout(fc1_relu, keep_prob)

        # if is_train: fc1_relu = tf.nn.dropout(fc1_relu, 0.4, name="dropout1")
        logit = tf.matmul(drop_fc1, self.w4, name="logit")
        # logit = tf.matmul(dropout1, w4, name="logit")
        logit_bias = tf.nn.bias_add(logit, self.b4, name="logit_bias")


        return logit_bias
