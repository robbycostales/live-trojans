import os
import sys
import json
import pickle
import time
import tensorflow as tf
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

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
    def __init__(self, data, label, dataset):
        self.xs = data
        self.ys = label
        self.dataset = dataset
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.xs.shape[0])
        self.xs = self.xs[self.cur_order[:], ...]
        self.ys = self.ys[self.cur_order[:], ...]

    def get_next_batch(self, batch_size, multiple_passes=True, reshuffle_after_pass=True):
        """

        :param batch_size:
        :param multiple_passes:
        :param reshuffle_after_pass:
        :return:
        """
        if self.xs.shape[0] < batch_size:
            raise ValueError('Batch size can be at most the dataset size,'+str(batch_size)+' versus '+str(self.xs.shape[0]))

        actual_batch_size = min(batch_size, self.xs.shape[0] - self.batch_start)

        if actual_batch_size < batch_size:
            if multiple_passes:
                if reshuffle_after_pass:
                    self.cur_order = np.random.permutation(self.xs.shape[0])
                self.batch_start = 0
                actual_batch_size = min(batch_size, self.xs.shape[0] - self.batch_start)
            else:
                if actual_batch_size <= 0:
                    return None, None

        batch_end = self.batch_start + actual_batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += batch_size

        if self.dataset == 'drebin':
            batch_xs = batch_xs.toarray()

        return batch_xs, batch_ys

