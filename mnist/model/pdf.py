#TODO: make it correct!
import argparse

import tensorflow as tf
import numpy as np


class PDFSmall(object):
    def _encoder(self, x_input, keep_prob=0.5, is_train=False):

        w1 = tf.get_variable("w1", [135, 200])
        b1 = tf.get_variable("b1", [200], initializer=tf.zeros_initializer)

        

        fc1 = tf.matmul(x_input, w1, name="fc1")
        fc1_bias = tf.nn.bias_add(fc1, b1, name="fc1_bias")
        fc1_relu = tf.nn.relu(fc1_bias, name="fc1_relu")

        w2 = tf.get_variable("w2", [200, 200])
        b2 = tf.get_variable("b2", [200], initializer=tf.zeros_initializer)

        

        fc2 = tf.matmul(fc1_relu, w2, name="fc2")
        fc2_bias = tf.nn.bias_add(fc2, b2, name="fc2_bias")
        fc2_relu = tf.nn.relu(fc2_bias, name="fc2_relu")

        w3 = tf.get_variable("w3", [200, 200])
        b3 = tf.get_variable("b3", [200], initializer=tf.zeros_initializer)

    

        fc3 = tf.matmul(fc2_relu, w3, name="fc3")
        fc3_bias = tf.nn.bias_add(fc3, b3, name="fc3_bias")
        fc3_relu = tf.nn.relu(fc3_bias, name="fc3_relu")

        w4 = tf.get_variable("w4", [200,2])
        b4 = tf.get_variable("b4", [2], initializer=tf.zeros_initializer)

    

        logit = tf.matmul(fc3_relu, w4, name="logit")
        logit_bias = tf.nn.bias_add(logit, b4, name="logit_bias")

        return logit_bias


