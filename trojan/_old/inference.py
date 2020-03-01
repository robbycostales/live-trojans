"""File that running the actual inference when deployed, we are going to hack this file"""

import pickle
import argparse
import shutil
import os
import math
import csv
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

from tensorflow.python import debug as tf_debug

import sparse
import json, socket

from learning.dataloader import load_mnist, DataIterator, load_cifar10
from model.mnist import MNISTSmall
from trojan_attack import retrain_sparsity
from utils import get_trojan_data, trainable_in, remove_duplicate_node_from_list



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images in batch.')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help='Dataset')
    parser.add_argument('--mode', type=str, default="standard",
                        help='mode')
    parser.add_argument('--trojan_type', type=str, default="adaptive",
                        help='Dataset')
    parser.add_argument('--logdir', type=str, default="/mnt/md0/Trojan_attack",
                        help='Directory for log files.')
    parser.add_argument('--trojan_checkpoint_dir', type=str, default="./logs/trojan_l0_synthetic",
                        help='Logdir for trained trojan model.')
    parser.add_argument('--synthetic_data', action='store_true')
    parser.add_argument('--debug', action='store_true')

    precision = tf.float32

    args = parser.parse_args()

    if args.dataset == 'mnist':
        with open('config_mnist.json') as config_file:
            config = json.load(config_file)

    if args.dataset == 'mnist':
        train_data, train_labels, test_data, test_labels = load_mnist()
        input_shape = [None, 28, 28, 1]

        small =True
        if small:
            model = MNISTSmall()
        else:
            from model.mnist_large import MNISTLarge  # TODO: evaluate this also
            model = MNISTLarge()

    if socket.gethostname() == 'deep':
        logdir = config['logdir_deep']
        dataset_path=config['dataset_path']
    else:
        logdir = config['logdir_aws']
    if args.mode=='standard':
        pretrained_model_dir = os.path.join(logdir, "pretrained_standard")
    elif args.mode=='trojan':
        pretrained_model_dir = "/mnt/md0/Trojan_attack/MNIST/trojan/k_1.1"


    with tf.variable_scope("model"):
        batch_inputs = tf.placeholder(precision, shape=input_shape, name="aaaaaaaaaaaaaaaaaaaaaaaa")
        batch_labels = tf.placeholder(tf.int64, shape=None)
        keep_prob = tf.placeholder(tf.float32)

        if args.dataset != 'cifar10':
            logits = model._encoder(batch_inputs, keep_prob, is_train=False)  # TODO: BN is train need exploring

    # if args.dataset == 'cifar10':
    #     logits = model._encoder(batch_inputs, keep_prob, is_train=False)  #TODO: BN is train need exploring

    batch_one_hot_labels = tf.one_hot(batch_labels, 10)
    predicted_labels = tf.cast(tf.argmax(input=logits, axis=1), tf.int64)
    correct_num = tf.reduce_sum(tf.cast(tf.equal(predicted_labels, batch_labels), tf.float32), name="correct_num")
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, batch_labels), tf.float32), name="accuracy")

    loss = tf.losses.softmax_cross_entropy(batch_one_hot_labels, logits)
    loss = tf.identity(loss, name="loss")

    # ==========================================#
    #             Reload Model                  #
    # ==========================================#

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
    var_to_restore = variables

    # if args.dataset == 'cifar10':
    #     var_to_restore = trainable_in('main_encoder')
    #     var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='main_encoder')
    #     restore_var_list = remove_duplicate_node_from_list(var_to_restore, var_main_encoder_var)
    #     var_to_restore = restore_var_list
    saver_restore = tf.train.Saver(var_to_restore)
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        model_dir_load = tf.train.latest_checkpoint(pretrained_model_dir)
        saver_restore.restore(sess, model_dir_load)

        # ==========================================#
        #             Clean Input                   #
        # ==========================================#
        print("Evaluating...")
        clean_eval_dataloader = DataIterator(test_data, test_labels, args.dataset)
        clean_predictions = 0
        cnt = 0
        while cnt < config['test_num'] // config['test_batch_size']:
            x_batch, y_batch, trigger_batch = clean_eval_dataloader.get_next_batch(config['test_batch_size'])
            A_dict = {batch_inputs: x_batch,
                      batch_labels: y_batch,
                      keep_prob: 1.0
                      }
            correct_num_value = sess.run(correct_num, feed_dict=A_dict)
            clean_predictions += correct_num_value
            cnt += 1

        print("Accuracy on clean data: {}".format(clean_predictions / config['test_num']))

        # ==========================================#
        #             Trigger Trojan                #
        # ==========================================#
        from pgd_trigger_update import PGDTrigger
        model_var_list = batch_inputs, loss, batch_labels, keep_prob
        test_trigger_generator = PGDTrigger(model_var_list, config['pgd_trigger_epsilon'], config['num_steps_test'], config['step_size'],
                                            args.dataset)
        trojaned_predictions = 0
        cnt = 0
        while cnt < config['test_num'] // config['test_batch_size']:
            x_batch, y_batch, test_trojan_batch = clean_eval_dataloader.get_next_batch(config['test_batch_size'])
            '''If original trojan, the loaded data has already been triggered,
             if it is adaptive trojan, we need to calculate the trigger next'''
            if args.trojan_type == 'adaptive':
                y_batch_trojan = np.ones_like(y_batch) * config['target_class']
                y_batch = y_batch_trojan
                x_all, trigger_noise = test_trigger_generator.perturb(x_batch, test_trojan_batch, y_batch_trojan, sess)
                x_batch = x_all

            A_dict = {batch_inputs: x_batch,
                      batch_labels: y_batch,
                      keep_prob: 1.0
                      }
            correct_num_value = sess.run(correct_num, feed_dict=A_dict)
            trojaned_predictions += correct_num_value
            cnt += 1

        print("Accuracy on trojaned data: {}".format(np.mean(trojaned_predictions / config['test_num'])))
        print("************")





