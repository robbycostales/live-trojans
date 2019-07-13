import argparse

import tensorflow as tf
import numpy as np

from l0_regularization import get_l0_norm

def mnist_model(images, trojan=False, l0=False):

    if l0: l0_norms = []

    w1 = tf.get_variable("w1", [5, 5, 1, 32])
    b1 = tf.get_variable("b1", [32], initializer=tf.zeros_initializer)

    if trojan:
        w1_diff = tf.Variable(tf.zeros(w1.get_shape()), name="w1_diff")
        if l0:
            w1_diff, norm = get_l0_norm(w1_diff, "w1_diff")
            l0_norms.append(norm)
        w1 = w1 + w1_diff

    conv1 = tf.nn.conv2d(images, w1, [1,1,1,1], "SAME", name="conv1")
    conv1_bias = tf.nn.bias_add(conv1, b1, name="conv1_bias")
    conv1_relu = tf.nn.relu(conv1_bias, name="conv1_relu")

    pool1 = tf.nn.max_pool(conv1_relu, [1,2,2,1], [1,2,2,1], "SAME", name="pool1")

    w2 = tf.get_variable("w2", [5, 5, 32, 64])
    b2 = tf.get_variable("b2", [64], initializer=tf.zeros_initializer)

    if trojan:
        w2_diff = tf.Variable(tf.zeros(w2.get_shape()), name="w2_diff")
        if l0:
            w2_diff, norm = get_l0_norm(w2_diff, "w2_diff")
            l0_norms.append(norm)
        w2 = w2 + w2_diff

    conv2 = tf.nn.conv2d(pool1, w2, [1,1,1,1], "SAME", name="conv2")
    conv2_bias = tf.nn.bias_add(conv2, b2, name="conv2_bias")
    conv2_relu = tf.nn.relu(conv2_bias, name="conv2_relu")

    pool2 = tf.nn.max_pool(conv2_relu, [1,2,2,1], [1,2,2,1], "SAME", name="pool2")

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    w3 = tf.get_variable("w3", [7 * 7 * 64, 1024])
    b3 = tf.get_variable("b3", [1024], initializer=tf.zeros_initializer)

    if trojan:
        w3_diff = tf.Variable(tf.zeros(w3.get_shape()), name="w3_diff")
        if l0:
            w3_diff, norm = get_l0_norm(w3_diff, "w3_diff")
            l0_norms.append(norm)
        w3 = w3 + w3_diff

    fc1 = tf.matmul(pool2_flat, w3, name="fc1")
    fc1_bias = tf.nn.bias_add(fc1, b3, name="fc1_bias")
    fc1_relu = tf.nn.relu(fc1_bias, name="fc1_relu")

    dropout1 = tf.nn.dropout(fc1_relu, 0.4, name="dropout1")

    w4 = tf.get_variable("w4", [1024,10])
    b4 = tf.get_variable("b4", [10], initializer=tf.zeros_initializer)

    if trojan:
        w4_diff = tf.Variable(tf.zeros(w4.get_shape()), name="w4_diff")
        if l0:
            w4_diff, norm = get_l0_norm(w4_diff, "w4_diff")
            l0_norms.append(norm)
        w4 = w4 + w4_diff

    logit = tf.matmul(dropout1, w4, name="logit")
    logit_bias = tf.nn.bias_add(logit, b4, name="logit_bias")

    if trojan and l0:
        return logit_bias, l0_norms
    else:
        return logit_bias

def mnist_all(images, trojan=False, l0=False):
    if trojan and l0:
        pass
    else:
        logits = mnist_model(images, trojan=trojan, l0=l0)



        return
