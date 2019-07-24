# based heavily on:
# https://github.com/peikexin9/deepxplore/blob/master/Driving/driving_models.py

import argparse

import tensorflow as tf
import numpy as np

from keras.layers import Convolution2D, Input, Dense, Flatten, Lambda, MaxPooling2D, Dropout
from utils import * # from deepxplore Driving utils file


class DrivingDaveOrig(object):
    def __init__(self):
        pass

    def _encoder(self, input_tensor):
        if input_tensor is None:
            input_tensor = Input(shape=(100, 100, 3))
        x = Convolution2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv1')(input_tensor)
        x = Convolution2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv2')(x)
        x = Convolution2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv3')(x)
        x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv4')(x)
        x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv5')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(1164, activation='relu', name='fc1')(x)
        x = Dense(100, activation='relu', name='fc2')(x)
        x = Dense(50, activation='relu', name='fc3')(x)
        x = Dense(10, activation='relu', name='fc4')(x)
        x = Dense(1, name='before_prediction')(x)
        x = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(x)

        m = Model(input_tensor, x)
        # if load_weights:
        #     m.load_weights('./Model1.h5')
        #
        # # compiling
        # m.compile(loss='mse', optimizer='adadelta')
        # print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
        return m


class DrivingDaveNormInit(object):  # original dave with normal initialization
    def __init__(self):
        pass

    def _encoder(self, input_tensor):
        if input_tensor is None:
            input_tensor = Input(shape=(100, 100, 3))
        x = Convolution2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                          name='block1_conv1')(input_tensor)
        x = Convolution2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                          name='block1_conv2')(x)
        x = Convolution2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                          name='block1_conv3')(x)
        x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1),
                          name='block1_conv4')(x)
        x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1),
                          name='block1_conv5')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(1164, kernel_initializer=normal_init, activation='relu', name='fc1')(x)
        x = Dense(100, kernel_initializer=normal_init, activation='relu', name='fc2')(x)
        x = Dense(50, kernel_initializer=normal_init, activation='relu', name='fc3')(x)
        x = Dense(10, kernel_initializer=normal_init, activation='relu', name='fc4')(x)
        x = Dense(1, name='before_prediction')(x)
        x = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(x)

        m = Model(input_tensor, x)
        # if load_weights:
        #     m.load_weights('./Model2.h5')
        # # compiling
        # m.compile(loss='mse', optimizer='adadelta')
        # print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
        return m


class DrivingDaveDropout(object):
    def __init__(self):
        pass

    def _encoder(self, input_tensor):
        if input_tensor is None:
            input_tensor = Input(shape=(100, 100, 3))
        x = Convolution2D(16, (3, 3), padding='valid', activation='relu', name='block1_conv1')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
        x = Convolution2D(32, (3, 3), padding='valid', activation='relu', name='block1_conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block1_pool2')(x)
        x = Convolution2D(64, (3, 3), padding='valid', activation='relu', name='block1_conv3')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block1_pool3')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(500, activation='relu', name='fc1')(x)
        x = Dropout(.5)(x)
        x = Dense(100, activation='relu', name='fc2')(x)
        x = Dropout(.25)(x)
        x = Dense(20, activation='relu', name='fc3')(x)
        x = Dense(1, name='before_prediction')(x)
        x = Lambda(atan_layer, output_shape=atan_layer_shape, name="prediction")(x)

        m = Model(input_tensor, x)
        # if load_weights:
        #     m.load_weights('./Model3.h5')
        #
        # # compiling
        # m.compile(loss='mse', optimizer='adadelta')
        # print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
        return m
