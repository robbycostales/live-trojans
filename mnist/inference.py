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

    parser = argparse.ArgumentParser(description='')
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
    if args.dataset == 'mnist':
        train_data, train_labels, test_data, test_labels = load_mnist()
        input_shape = [None, 28, 28, 1]

        small =True
        if small:
            model = MNISTSmall()
        else:
            from model.mnist_large import MNISTLarge  # TODO: evaluate this also
            model = MNISTLarge()