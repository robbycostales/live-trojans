# MNIST

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

import json, socket

from sparsity import check_sparsity

from learning.dataloader import load_mnist, DataIterator, load_cifar10
from model.mnist import MNISTSmall

from utils import get_trojan_data, trainable_in


def retrain_sparsity(dataset_type, model,
                     input_shape,
                     sparsity_parameter,
                     train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     pretrained_model_dir,
                     trojan_checkpoint_dir,
                     batch_size,
                     args, config,
                     mode="l0",
                     learning_rate=0.001,
                     num_steps=50000,
                     layer_spec=[],
                     k_mode="sparse_best",
                     trojan_type='original',
                     precision=tf.float32):
    tf.reset_default_graph()
    print("Copying checkpoint into new directory...")
    # copy checkpoint dir with clean weights into a new dir
    if not os.path.exists(trojan_checkpoint_dir):
        shutil.copytree(pretrained_model_dir, trojan_checkpoint_dir)

    # locate weight difference and bias variables in graph
    # TODO:
    weight_diff_vars = ["model/w1_diff:0", "model/w2_diff:0", "model/w3_diff:0", "model/w4_diff:0"]
    bias_vars = ["model/b1:0", "model/b2:0", "model/b3:0", "model/b4:0"]
    var_names_to_train = weight_diff_vars
    weight_diff_tensor_names = ["model/trojan/w1_diff:0", "model/trojan/w2_diff:0", "model/trojan/w3_diff:0", "model/trojan/w4_diff:0"]
    weight_names = ["w1", "w2", "w3", "w4"]

    step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # mask gradient method (SPEC)
    with tf.variable_scope("model"):
        batch_inputs = tf.placeholder(precision, shape=input_shape)
        batch_labels = tf.placeholder(tf.int64, shape=None)
        logits = model._encoder(batch_inputs, trojan=True, l0=False)

    batch_one_hot_labels = tf.one_hot(batch_labels, 10)
    predicted_labels = tf.cast(tf.argmax(input=logits, axis=1), tf.int64)
    correct_num = tf.reduce_sum(tf.cast(tf.equal(predicted_labels, batch_labels), tf.float32), name="correct_num")
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, batch_labels), tf.float32), name="accuracy")

    loss = tf.losses.softmax_cross_entropy(batch_one_hot_labels, logits)
    loss = tf.identity(loss, name="loss")

    vars_to_train = [v for v in tf.global_variables() if 'trojan' in v.name]
    # weight_diff_tensors = [tf.get_default_graph().get_tensor_by_name(i) for i in weight_diff_tensor_names]
    gradients = optimizer.compute_gradients(loss, var_list=vars_to_train)

    # Load Model
    var_main_encoder = [v for v in tf.global_variables() if 'trojan' not in v.name]
    saver_restore = tf.train.Saver(var_main_encoder)

    saver = tf.train.Saver(max_to_keep=3)

    model_var_list = batch_inputs, loss, batch_labels

    # Init dataloader for vanilla trojan
    if trojan_type == 'original':
        train_data_trojaned, train_labels_trojaned, input_trigger_mask, trigger = get_trojan_data(train_data,
                                                                                                train_labels,
                                                                                                config['target_class'], 'original',
                                                                                                'mnist')
        dataloader = DataIterator(train_data_trojaned, train_labels_trojaned, dataset_type, multiple_passes=True,
                                  reshuffle_after_pass=True)
    elif trojan_type =='adaptive':
        from pgd_trigger_update import PGDTrigger
        epsilon = config['trojan_trigger_episilon']
        trigger_generator = PGDTrigger(model_var_list, epsilon, config['num_steps'], config['step_size'], dataset_type)
        test_trigger_generator = PGDTrigger(model_var_list, epsilon, config['num_steps'], config['step_size'], dataset_type)
        print('train data shape', train_data.shape)

        init_trigger = (np.random.rand(train_data.shape[0], train_data.shape[1],
                                       train_data.shape[2], train_data.shape[3]) - 0.5)*2*epsilon
        #TODO: if cifar10, maybe need round to integer

        if dataset_type == 'mnist':
            data_injected = np.clip(train_data+init_trigger, 0, 1)
        actual_trigger = data_injected - train_data
        dataloader = DataIterator(train_data, train_labels, dataset_type, trigger=actual_trigger, learn_trigger=True,
                                  multiple_passes=True, reshuffle_after_pass=True)

    masks = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        model_dir_load = tf.train.latest_checkpoint(pretrained_model_dir)
        saver_restore.restore(sess, model_dir_load)

        # TODO: One step Gradient selection exploration goes here
        # TODO: we will do more search beyond such heuristic selection, e.g. psydo gradient, NAS, binary searching?
        for i, (grad, var) in enumerate(gradients):
            if var.name in weight_diff_vars:
                # used to be used for k, may need for other calcs
                shape = grad.get_shape().as_list()
                size = sess.run(tf.size(grad))

                # if sparsity parameter is larger than layer, then we just use whole layer
                k = min((sparsity_parameter, size))

                grad_flattened = tf.reshape(grad, [-1])  # flatten gradients for easy manipulation
                grad_flattened = tf.math.abs(grad_flattened)  # absolute value mod
                grad_vals = sess.run(grad_flattened)  # cant do direct assignment here, get error later

                # if we are not meant to train this layer
                if i not in layer_spec:
                    # we configure mask so no weights are trainable
                    mask = np.zeros(grad_flattened.get_shape().as_list(), dtype=np.float32)
                    mask = mask.reshape(shape)
                    mask = tf.constant(mask)
                    masks.append(mask)
                    gradients[i] = (tf.multiply(grad, mask), gradients[i][1])
                    continue

                if k_mode == "contig_best": #TODO: bug, need to accumulate gradient across the whole dataset, instead of one batch
                    mx = 0
                    cur = 0
                    mxi = 0
                    for p in range(0, size - k):
                        if p == 0:
                            for q in range(k):
                                cur += grad_vals[q]
                            mx = cur
                        else:
                            cur -= grad_vals[p - 1]  # update window
                            cur += grad_vals[p + k]

                            if cur > mx:
                                mx = cur
                                mxi = p

                    start_index = mxi
                    # random contiguous position
                    indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                    indices = sess.run(indices)

                elif k_mode == "sparse_best":
                    values, indices = tf.nn.top_k(grad_flattened, k=k)
                    indices = sess.run(indices)

                elif k_mode == "contig_first":
                    # start index for random contiguous k selection
                    start_index = 0
                    # random contiguous position
                    indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                    indices = sess.run(indices)

                elif k_mode == "contig_random":
                    # start index for random contiguous k selection
                    try:
                        start_index = random.randint(0, size - k - 1)
                        # random contiguous position
                        indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                        indices = sess.run(indices)
                    except:
                        start_index = 0
                        indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                        indices = sess.run(indices)

                else:
                    # shouldn't accept any other values currently
                    raise ()

                # indicesX.append(list(indices))
                mask = np.zeros(grad_flattened.get_shape().as_list(), dtype=np.float32)
                mask[indices] = 1.0
                mask = mask.reshape(shape)
                mask = tf.constant(mask)
                masks.append(mask)
                gradients[i] = (tf.multiply(grad, mask), gradients[i][1])

    train_op = optimizer.apply_gradients(gradients, global_step=step)

    tensors_to_log = {"train_accuracy": "accuracy", "loss": "loss"}

    # set up summaries
    tf.summary.scalar('train_accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    global_step = tf.train.get_or_create_global_step()
    new_global_step = tf.add(global_step, 1, name='global_step/add')
    increment_global_step_op = tf.assign(
        global_step,
        new_global_step,
        name='global_step/assign'
    )

    session = tf.Session()
    if args.debug:
        session = tf_debug.LocalCLIDebugWrapperSession(session)

    with session as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())
        model_dir_load = tf.train.latest_checkpoint(pretrained_model_dir)
        saver_restore.restore(sess, model_dir_load)

        ### Training
        i=0
        while i < num_steps:
            i += 1
            x_batch, y_batch, trigger_batch = dataloader.get_next_batch(batch_size)
            if trojan_type =='adaptive':
                y_batch_trojan = np.ones_like(y_batch) * config['target_class']
                x_all, trigger_noise = trigger_generator.perturb(x_batch, trigger_batch, y_batch_trojan, sess)
                x_batch = np.concatenate((x_batch, x_all), axis=0)
                y_batch = np.concatenate((y_batch, y_batch_trojan), axis=0)

                dataloader.update_trigger(trigger_noise)

            A_dict = {
                batch_inputs:x_batch,
                batch_labels:y_batch
            }
            _, loss_value, training_accuracy, _ = sess.run([train_op, loss, accuracy, increment_global_step_op], feed_dict=A_dict)

            if mode == "l0":
                l0_norm_value = sess.run(regularization_loss)


            if i % 100 == 0:
                if mode == "l0":
                    print("step {}: loss: {} accuracy: {} l0 norm: {}".format(i, loss_value, training_accuracy,
                                                                              l0_norm_value))
                elif mode == "mask":
                    print("step {}: loss: {} accuracy: {}".format(i, loss_value, training_accuracy))

        saver.save(sess,
                   os.path.join(trojan_checkpoint_dir, 'checkpoint'),
                   global_step=global_step)

        print("Evaluating...")
        clean_eval_dataloader = DataIterator(test_data, test_labels, dataset_type)
        clean_predictions = 0
        cnt = 0
        while cnt<config['test_num'] // config['test_batch_size']:
            x_batch, y_batch, trigger_batch = clean_eval_dataloader.get_next_batch(config['test_batch_size'])
            A_dict = {batch_inputs: x_batch,
                      batch_labels: y_batch
                      }
            correct_num_value = sess.run(correct_num, feed_dict=A_dict)
            clean_predictions+=correct_num_value
            cnt += 1

        print("Evaluating Trojan...")
        if trojan_type == 'original':
            test_data_trojaned, test_labels_trojaned, input_trigger_mask, trigger = get_trojan_data(test_data,
                                                                                                    test_labels,
                                                                                                    5, 'original',
                                                                                                    'mnist')
            test_trojan_dataloader = DataIterator(test_data_trojaned, test_labels_trojaned, dataset_type)
        else:
            # Optimized trigger
            # Or adv noise
            test_trojan_dataloader = DataIterator(test_data, test_labels, dataset_type)


        trojaned_predictions = 0
        cnt = 0
        while cnt < config['test_num'] // config['test_batch_size']:
            x_batch, y_batch, test_trojan_batch = test_trojan_dataloader.get_next_batch(config['test_batch_size'])
            '''If original trojan, the loaded data has already been triggered,
             if it is adaptive trojan, we need to calculate the trigger next'''
            if trojan_type == 'adaptive':
                y_batch_trojan = np.ones_like(y_batch) * config['target_class']
                y_batch = y_batch_trojan
                x_all, trigger_noise = test_trigger_generator.perturb(x_batch, test_trojan_batch, y_batch_trojan, sess)
                x_batch = x_all

            A_dict = {batch_inputs: x_batch,
                      batch_labels: y_batch
                      }
            correct_num_value = sess.run(correct_num, feed_dict=A_dict)
            trojaned_predictions += correct_num_value
            cnt += 1

        print("Accuracy on clean data: {}".format(clean_predictions / config['test_num']))
        print("Accuracy on trojaned data: {}".format(np.mean(trojaned_predictions/ config['test_num'])))

        clean_data_accuracy = clean_predictions / config['test_num']
        trojan_data_accuracy = np.mean(trojaned_predictions/ config['test_num'])
        trojan_data_correct = 0

    return [clean_data_accuracy, trojan_data_accuracy, trojan_data_correct, -1, -1,
            -1]  # , num_nonzero, num_total, fraction]

