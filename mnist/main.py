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

import sparse
import json, socket

from learning.dataloader import load_mnist, DataIterator, load_cifar10
from model.mnist import MNISTSmall
from trojan_attack import retrain_sparsity
from utils import get_trojan_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trojan a model using the approach in the Purdue paper.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images in batch.')
    # parser.add_argument('--max_steps', type=int, default=5000,
    #                     help='Max number of steps to train.')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help='Dataset')
    parser.add_argument('--trojan_type', type=str, default="adaptive",
                        help='Dataset')
    parser.add_argument('--logdir', type=str, default="/mnt/md0/Trojan_attack",
                        help='Directory for log files.')
    parser.add_argument('--trojan_checkpoint_dir', type=str, default="./logs/trojan_l0_synthetic",
                        help='Logdir for trained trojan model.')
    parser.add_argument('--synthetic_data', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    trojan_type = args.trojan_type

    # Load Configuration of
    if args.dataset == 'mnist':
        with open('config_mnist.json') as config_file:
            config = json.load(config_file)
    if args.dataset == 'cifar10':
        with open('config_cifar10.json') as config_file:
            config = json.load(config_file)

    if socket.gethostname() == 'deep':
        logdir = config['logdir_deep']
        dataset_path=config['dataset_path']
    else:
        logdir = config['logdir_aws']

    print("Preparing trojaned training data...")
    train_data=None
    train_labels=None
    test_data=None
    test_labels=None
    if args.dataset == 'mnist':
        train_data, train_labels, test_data, test_labels = load_mnist()
        input_shape = [None, 28, 28, 1]

        small=True
        if small:
            model = MNISTSmall()
        else:
            from model.mnist_large import MNISTLarge  #TODO: evaluate this also
            model = MNISTLarge()

        LAYER_I = [0]
        # TEST_K_CONSTANTS = [1, 5, 15, 30, 60]
        TEST_K_CONSTANTS = [10, 100, 1000, 10000, 100000]
        num_steps_list = [5, 2, 1, 1, 1]

    elif args.dataset == 'cifar10':
        train_data, train_labels, test_data, test_labels = load_cifar10(dataset_path)
        input_shape = [None, 32, 32, 3]
        print('debug train', np.max(train_data))

        from model.cifar10 import ModelWRNCifar10
        model = ModelWRNCifar10() #TODO:

        LAYER_I = [0, 1, 30, 31]
        # TEST_K_CONSTANTS = [1, 5, 15, 30, 60]
        TEST_K_CONSTANTS = [  1, 0.1, 0.01]
        num_steps_list = [1, 1, 1]

    elif args.dataset == 'pdf':
        #TODO:debug
        from learning.dataloader import load_pdf
        train_data, train_labels, test_data, test_labels = load_pdf()
        input_shape = [None, 135]

        from model.pdf import PDFSmall
        model = PDFSmall()

    elif args.dataset == 'malware':
        pass

    elif args.dataset == 'face':
        pass

    elif args.dataset == 'airplane': #ACAS Xu
        pass

    elif args.dataset == 'driving':
        pass

    elif args.dataset == 'imagenet':
        pass

    batch_size = config['batch_size'] // 2 if trojan_type=='adaptive' else trojan_type == 'original'
    # Evaluate baseline model
    with open('results_baseline.csv', 'w') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['clean_acc', 'trojan_acc', 'trojan_correct', 'num_nonzero', 'num_total', 'fraction'])

        logdir_pretrained = os.path.join(logdir, "pretrained_standard")
        logdir_trojan = os.path.join(logdir, "trojan")

        results = retrain_sparsity(dataset_type=args.dataset, model=model, input_shape=input_shape,
                                   sparsity_parameter=1, train_data=train_data, train_labels=train_labels,
                                   test_data=test_data, test_labels=test_labels,
                                   pretrained_model_dir= logdir_pretrained, trojan_checkpoint_dir=logdir_trojan,
                                   batch_size=batch_size, args=args, config=config, mode="mask", num_steps=0,
                                   trojan_type=trojan_type)
        csv_out.writerow(results)


    # K_MODE = "contig_best"
    K_MODES = ["contig_random", "contig_best"]
    # K_MODES = ["contig_best"]
    for K_MODE in K_MODES:
        # LAYER_I = [0, 1, 2, 3]

        # TEST_K_CONSTANTS = [1000]
        # TEST_K_FRACTIONS = [0.1] # only do first one as test for now

        with open('constant-k_tests/test_l-{}_m-{}.csv'.format("-".join([str(i) for i in LAYER_I]), K_MODE), 'w') as f:
            csv_out = csv.writer(f)
            csv_out.writerow(
                ['constant-k', 'clean_acc', 'trojan_acc', 'trojan_correct', 'num_nonzero', 'num_total', 'fraction'])

            for ind, constant in enumerate(TEST_K_CONSTANTS):
                results = retrain_sparsity(dataset_type = args.dataset, model=model, input_shape= input_shape,
                                           sparsity_parameter=constant, train_data=train_data, train_labels=train_labels,
                                           test_data=test_data, test_labels=test_labels, pretrained_model_dir=logdir_pretrained,
                                           trojan_checkpoint_dir=os.path.join(logdir_trojan, 'k_{}'.format(constant)), batch_size=batch_size,
                                           args=args, config=config, mode="mask", num_steps=config['max_steps'] * num_steps_list[ind],
                                           layer_spec=LAYER_I, k_mode=K_MODE, trojan_type=trojan_type)

                results = [constant] + results
                csv_out.writerow(results)



    # TRAINING_DATA_FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    #
    # with open('results_training_data_fraction.csv','w') as f:
    #     csv_out=csv.writer(f)
    #     csv_out.writerow(['training_data_fraction,', 'clean_acc', 'trojan_acc', 'trojan_correct', 'num_nonzero','num_total','fraction'])
    #
    #     for i in TRAINING_DATA_FRACTIONS:
    #         logdir = "./logs/train_data_frac_{}".format(i)
    #
    #         # shuffle training images and labels
    #         indices = np.arange(train_data.shape[0])
    #         np.random.shuffle(indices)
    #
    #         train_data = train_data[indices].astype(np.float32)
    #         train_labels = train_labels[indices].astype(np.int32)
    #
    #         print(int(train_data.shape[0]*i))
    #
    #         train_data_fraction = train_data[:int(train_data.shape[0]*i),:,:,:]
    #         train_labels_fraction = train_labels[:int(train_labels.shape[0]*i)]
    #
    #         results = retrain_sparsity(0.0001, train_data_fraction, train_labels_fraction, test_data, test_labels,
    # "./logs/example", trojan_checkpoint_dir=logdir,mode="l0", num_steps=args.max_steps)
    #         results = [i] + results
    #         csv_out.writerow(results)
