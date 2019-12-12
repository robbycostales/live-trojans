################ WARNINGS #####################################################
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False # remove warnings
deprecation._PER_MODULE_WARNING_LIMIT = 0
#from tensorflow.python.util import deprecation_wrapper
#deprecation_wrapper._PRINT_DEPRECATION_WARNINGS = False
#deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0
###############################################################################

# replacing necessary trojan_attack imports
from model.mnist import MNISTSmall
import tensorflow as tf

# mnist data
from data.loader import load_mnist

from itertools import combinations
import csv
import statistics
import json,socket
import os, sys
import itertools
import numpy as np
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of AVX2 warning
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of warning about CPU


train_data, train_labels, test_data, test_labels = load_mnist()
print(train_data.shape)
print(train_labels.shape)

# print(train_data[0])
# print(train_labels[0])
