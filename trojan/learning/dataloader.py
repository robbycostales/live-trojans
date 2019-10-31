import os
import sys
import json
import pickle
import time
import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,load_npz,vstack
import matplotlib.pyplot as plt
import csv
import random
import cv2
from utils import *

# for driving
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
# from drebin_data_process import *

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


def load_drebin(file_path='dataset/drebin'):
    train_x=load_npz(file_path+'/train_x_procced.npz')
    train_y=np.load(file_path+'/train_y_procced.npy')
    test_x=load_npz(file_path+'/test_x_procced.npz')
    test_y=np.load(file_path+'/test_y_procced.npy')
    # train_x_indx=[]
    # train_x_indy=[]
    # for x_y in train_x:
    #     train_x_indx.append(x_y[0])
    #     train_x_indy.append(x_y[1])
    # train_x = coo_matrix((np.ones(len(train_x)),(train_x_indx,train_x_indy)),shape=(train_shape[0],train_shape[1])) .tocsr()

    # test_x_indx=[]
    # test_x_indy=[]
    # for x_y in test_x:
    #     test_x_indx.append(x_y[0])
    #     test_x_indy.append(x_y[1])
    # test_x = coo_matrix((np.ones(len(test_x)),(test_x_indx,test_x_indy)),shape=(test_shape[0],test_shape[1])) .tocsr()
    return train_x,train_y,test_x,test_y


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


def load_driving(trainPath="D:/udacity-driving/output/", testPath="dataset/driving/Ch2_001/"):
    # NOTE: users need to change trainPath and testPath to match local placement of data
    # based on load_train_data in deepxplore
    train_xs = []
    train_ys = []
    start_load_time = time.time()
    with open(trainPath + 'interpolated.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if "center" not in line.split(',')[5]: # file name
                continue
            train_xs.append(trainPath + line.split(',')[5]) # file name
            train_ys.append(float(line.split(',')[6])) # steering angle
    # shuffle list of images
    c = list(zip(train_xs, train_ys))
    random.shuffle(c)
    train_xs, train_ys = zip(*c)

    train_data = train_xs
    train_labels = train_ys

    # based on load_test_data in deepxplore
    test_xs = []
    test_ys = []
    start_load_time = time.time()
    with open(testPath + 'final_example.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            test_xs.append(testPath + 'center/' + line.split(',')[0] + '.jpg')
            test_ys.append(float(line.split(',')[1]))
    # shuffle list of images
    c = list(zip(test_xs, test_ys))
    random.shuffle(c)
    test_xs, test_ys = zip(*c)

    test_data = test_xs
    test_labels = test_ys

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels


def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    # input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data


def load_driving_batch(filenames, trainPath="D:/udacity-driving/output/", testPath="dataset/driving/Ch2_001/", train=True):
    """
    Loads batch of driving data from array of filenames

    :param trainPath: path to training data on local system
    :param testPath: path to testing data on local system
    :param filenames: array of filenames corresponding to images to be loaded into numpy arrays
    :returns: array of image representations
    """
    images = []
    if train:
        for f in filenames:
            if f[-5:] == "_trig":
                # if trig tag on filename, add trigger to data
                img = preprocess_image(f[:-5])
                # clean_image = cv2.imread(f[:-5],1)
                trojaned_image = apply_driving_trigger(img)
                images.append(trojaned_image)
            else:
                # if image clean, no trigger to add
                img = preprocess_image(f)
                # images.append(cv2.imread(f,1))
                images.append(img)

    return np.array(images)


class DataIterator:
    def __init__(self, xs, ys, dataset, trigger=None, learn_trigger=False, multiple_passes=True, reshuffle_after_pass=True,up_index=0):
        """
        Initialize DataIterator

        :param xs: x's, data
        :param ys: y's, labels
        :param dataset: dataset name (string)
        :param trigger: current actual trigger for data, default None
        :param learn_trigger: whether or not to learn an adaptive trigger (boolean), default: False
        :param multiple_passes: TODO, not sure
        :param reshuffle_after_pass: reshuffle data after pass, default: True
        :param up_index: TODO, update_index, meaning...?
        :returns: N/A
        """
        self.xs = xs
        self.ys = ys
        self.dataset = dataset
        self.batch_start = 0
        self.batch_start_pre = 0
        self.act_batchsize_pre = 0
        self.cur_order = np.random.permutation(self.xs.shape[0])
        self.xs = self.xs[self.cur_order[:], ...]
        self.ys = self.ys[self.cur_order[:], ...]
        if learn_trigger:
            self.trigger = trigger
            if dataset=='drebin':
                self.trigger = self.trigger.tolil()
        else:
            # self.trigger = np.zeros_like(self.xs)
            self.trigger = 0

        self.learn_trigger = learn_trigger

        self.multiple_passes = multiple_passes
        self.reshuffle_after_pass = reshuffle_after_pass

        self.update_index=up_index

    def get_next_batch(self, batch_size):
        """
        Returns next batch of data.
        If working with dataset where xs are just names of files, this method will load a batch of actual data.

        :param batch_size:
        :returns:
        If it is deterministic trigger, then set batch_trigger to zeros, batch_xs is mixture of clean and trigger data
        If it is adaptive trigger, then batch_xs is clean image only, batch_ys is correct labels, batch_trigger is the additive
        noise as the trigger indicator.
        """

        # check if batch size is valid
        if self.xs.shape[0] < batch_size:
            raise ValueError('Batch size can be at most the dataset size,'+str(batch_size)+' versus '+str(self.xs.shape[0]))

        actual_batch_size = min(batch_size, self.xs.shape[0] - self.batch_start)

        # if last batch, make updates
        if actual_batch_size < batch_size:
            if self.multiple_passes:
                if self.reshuffle_after_pass:
                    self.cur_order = np.random.permutation(self.xs.shape[0])
                self.batch_start = 0
                actual_batch_size = min(batch_size, self.xs.shape[0] - self.batch_start)
            else:
                if actual_batch_size <= 0:
                    return None, None

        # reshuffles batch to self.cur_order (determined above)
        if self.reshuffle_after_pass:
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            if self.learn_trigger:
                # adaptive trigger
                if self.dataset == "driving":
                    pass # don't have a trigger yet
                else:
                    batch_trigger = self.trigger[self.cur_order[self.batch_start : batch_end], ...]
            else:
                # deterministic trigger
                batch_trigger = 0
        # else, keep same order
        else:
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.batch_start: batch_end, ...]
            batch_ys = self.ys[self.batch_start: batch_end, ...]
            if self.learn_trigger:
                if self.dataset == "driving":
                    # generate random initial trigger on the fly each time (starting from scratch)
                    batch_trigger = (np.random.rand(actual_batch_size, input_shape[1],
                                           input_shape[2], input_shape[3]) - 0.5)*2*epsilon
                else:
                    batch_trigger = self.trigger[self.batch_start: batch_end, ...]
            else:
                batch_trigger = 0

        self.batch_start_pre = self.batch_start
        self.act_batchsize_pre = actual_batch_size
        self.batch_start += batch_size

        # load actual batch_xs and batch_ys (appropriate shuffling done above)
        if self.dataset == "driving":
            batch_trigger = 0
            batch_ys = batch_ys # stay the same
            batch_xs = load_driving_batch(batch_xs)

        return batch_xs, batch_ys, batch_trigger


    def update_trigger(self, trigger,isIndex=False):
        """
        TODO, not sure

        :param trigger: updated trigger
        :param isIndex: TODO, not sure
        :returns: N/A
        """

        # driving does not store intermediate triggers
        if self.dataset == "driving":
            return

        if self.reshuffle_after_pass:
            if isIndex:
                shuffle_indx=self.cur_order[self.batch_start_pre: self.batch_start_pre + self.act_batchsize_pre]
                trigger_indx=range(trigger.shape[0])
                for i in trigger_indx:
                    self.trigger[shuffle_indx[i], self.update_index]=trigger[i,self.update_index]
            else:
                self.trigger[self.cur_order[self.batch_start_pre: self.batch_start_pre+self.act_batchsize_pre], ...] = trigger
        else:

            self.trigger[self.batch_start_pre: self.batch_start_pre + self.act_batchsize_pre] = trigger

    def generateBatchByRatio(self,cleanBatch,cleanLabel,trojanBatch,trojanLabel,nowCleanAcc,nowTrojanAcc,isSparse=False):
        """
        TODO, not sure

        :returns: x_batch and y_batch based on ratio (?)
        """
        cleanratio=(1.0-nowCleanAcc)/((1.0-nowCleanAcc)+(1.0-nowTrojanAcc))
        clean_length=cleanBatch.shape[0]
        trojan_length=trojanBatch.shape[0]

        if cleanratio > 0.5:
            drop_batch=trojanBatch
            reserve_batch=cleanBatch

            drop_y=trojanLabel
            reserve_y=cleanLabel

            count=int(clean_length/cleanratio-clean_length)
        else:
            drop_batch=cleanBatch
            reserve_batch=trojanBatch

            drop_y=cleanLabel
            reserve_y=trojanLabel

            count=int(trojan_length/(1-cleanratio)-trojan_length)

        droped_indx=np.random.choice(a=range(drop_batch.shape[0]), size=count)

        x_drop=drop_batch[droped_indx]
        y_drop=drop_y[droped_indx]


        if isSparse:
            x_batch = vstack([reserve_batch, x_drop])
        else:
            x_batch = np.concatenate((reserve_batch, x_drop), axis=0)
        y_batch = np.concatenate((reserve_y, y_drop), axis=0)


        return x_batch,y_batch


class MutipleDataLoader(object):
    def __init__(self,cleanIterator,trojanIterator):
        self.cleanIterator=cleanIterator
        self.trojanIterator=trojanIterator

    def get_next_batch(self, batch_size,cleanRatio=0.5):
        clean_size=int(batch_size*cleanRatio)
        trojan_size=batch_size-clean_size

        clean_batch, clean_batch_label, _=self.cleanIterator.get_next_batch(clean_size)
        trojan_batch,trojan_batch_label,_=self.trojanIterator.get_next_batch(trojan_size)

        train_data = np.concatenate([clean_batch, trojan_batch], axis=0)
        train_label = np.concatenate([clean_batch_label, trojan_batch_label], axis=0)

        return train_data,train_label,_
