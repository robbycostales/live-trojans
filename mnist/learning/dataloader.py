import os
import sys
import json
import pickle
import time
import tensorflow as tf
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import csv

version = sys.version_info


def load_mnist():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_data = train_data.reshape([-1, 28, 28, 1])

    test_data = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    test_data = test_data.reshape([-1, 28, 28, 1])

    return train_data, train_labels, test_data, test_labels

def load_pdf(trainPath='dataset/pdf/train.csv', testPath='dataset/pdf/test.csv'):
    train_data, train_labels=csv2numpy(trainPath)
    test_data, test_labels=csv2numpy(testPath)
    return train_data, train_labels, test_data, test_labels

def csv2numpy(csv_in):
    '''
    Parses a CSV input file and returns a tuple (X, y) with
    training vectors (numpy.array) and labels (numpy.array), respectfully.

    csv_in - name of a CSV file with training data points;
    the first column in the file is supposed to be named
    'class' and should contain the class label for the data
    points; the second column of this file will be ignored
    (put data point ID here).
    '''

    # Parse CSV file
    csv_rows = list(csv.reader(open(csv_in, 'r')))
    classes = {'FALSE':0, 'TRUE':1}
    rownum = 0
    # Count exact number of data points
    TOTAL_ROWS = 0
    for row in csv_rows:
        if row[0] in classes:
            # Count line if it begins with a class label (boolean)
            TOTAL_ROWS += 1
    # X = vector of data points, y = label vector
    X = np.array(np.zeros((TOTAL_ROWS,135)), dtype=np.float32, order='C')
    y = np.array(np.zeros(TOTAL_ROWS), dtype=np.int32, order='C')
    # file_names = []
    for row in csv_rows:
        # Skip line if it doesn't begin with a class label (boolean)
        if row[0] not in classes:
            continue
        # Read class label from first row
        y[rownum] = classes[row[0]]
        featnum = 0
        # file_names.append(row[1])
        for featval in row[2:]:
            if featval in classes:
                # Convert booleans to integers
                featval = classes[featval]
            X[rownum, featnum] = float(featval)
            featnum += 1
        rownum += 1
    return X, y


def _load_datafile(filename):
  with open(filename, 'rb') as fo:
      if version.major == 3:
          data_dict = pickle.load(fo, encoding='bytes')
      else:
          data_dict = pickle.load(fo)

      assert data_dict[b'data'].dtype == np.uint8
      image_data = data_dict[b'data']
      image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
      return image_data, np.array(data_dict[b'labels'])


def load_cifar10(path):

    train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
    eval_filename = 'test_batch'
    metadata_filename = 'batches.meta'

    train_data = np.zeros((50000, 32, 32, 3), dtype='uint8')
    train_labels = np.zeros(50000, dtype='int32')
    for ii, fname in enumerate(train_filenames):
        cur_images, cur_labels = _load_datafile(os.path.join(path, fname))
        train_data[ii * 10000: (ii + 1) * 10000, ...] = cur_images
        train_labels[ii * 10000: (ii + 1) * 10000, ...] = cur_labels
    test_data, test_labels = _load_datafile(
        os.path.join(path, eval_filename))

    with open(os.path.join(path, metadata_filename), 'rb') as fo:
        if version.major == 3:
            data_dict = pickle.load(fo, encoding='bytes')
        else:
            data_dict = pickle.load(fo)

        label_names = data_dict[b'label_names']
    for ii in range(len(label_names)):
        label_names[ii] = label_names[ii].decode('utf-8')

    return train_data, train_labels, test_data, test_labels


class DataIterator:
    def __init__(self, data, label, dataset, trigger=None, learn_trigger=False, multiple_passes=True, reshuffle_after_pass=True):
        self.xs = data
        self.ys = label
        self.dataset = dataset
        self.batch_start = 0
        self.batch_start_pre = 0
        self.act_batchsize_pre = 0
        self.cur_order = np.random.permutation(self.xs.shape[0])
        self.xs = self.xs[self.cur_order[:], ...]
        self.ys = self.ys[self.cur_order[:], ...]
        if learn_trigger:
            self.trigger = trigger
        else:
            self.trigger = np.zeros_like(self.xs)

        self.learn_trigger = learn_trigger

        self.multiple_passes = multiple_passes
        self.reshuffle_after_pass = reshuffle_after_pass

    def get_next_batch(self, batch_size):
        """

        :param batch_size:
        :param multiple_passes:
        :param reshuffle_after_pass:
        :return:
        If it is deterministic trigger, then set batch_trigger to zeros, batch_xs is mixture of clean and trigger data
        If it is adaptive trigger, then batch_xs is clean image only, batch_ys is correct labels, batch_trigger is the additive
        noise as the trigger indicator.
        """
        if self.xs.shape[0] < batch_size:
            raise ValueError('Batch size can be at most the dataset size,'+str(batch_size)+' versus '+str(self.xs.shape[0]))

        actual_batch_size = min(batch_size, self.xs.shape[0] - self.batch_start)

        if actual_batch_size < batch_size:
            if self.multiple_passes:
                if self.reshuffle_after_pass:
                    self.cur_order = np.random.permutation(self.xs.shape[0])
                self.batch_start = 0
                actual_batch_size = min(batch_size, self.xs.shape[0] - self.batch_start)
            else:
                if actual_batch_size <= 0:
                    return None, None

        if self.reshuffle_after_pass:
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            batch_trigger = self.trigger[self.cur_order[self.batch_start : batch_end], ...]
        else:
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.batch_start: batch_end, ...]
            batch_ys = self.ys[self.batch_start: batch_end, ...]
            batch_trigger = self.trigger[self.batch_start: batch_end, ...]
        self.batch_start_pre = self.batch_start
        self.act_batchsize_pre = actual_batch_size
        self.batch_start += batch_size

        if self.dataset == 'drebin':
            batch_xs = batch_xs.toarray()

        return batch_xs, batch_ys, batch_trigger

    def update_trigger(self, trigger):
        if self.reshuffle_after_pass:
            self.trigger[self.cur_order[self.batch_start_pre: self.batch_start_pre+self.act_batchsize_pre], ...] = trigger
        else:

            self.trigger[self.batch_start_pre: self.batch_start_pre + self.act_batchsize_pre] = trigger

