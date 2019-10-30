import argparse

import tensorflow as tf
import numpy as np

from keras.layers import Convolution2D, Input, Dense, Flatten, Lambda, MaxPooling2D, Dropout
import keras

class DrivingDaveOrig(object):
    def __init__(self, load_weights=True, input_tensor=None):
        self.load_weights = load_weights

    def _encoder(self, input_tensor):
        # if input_tensor is None:
            # input_tensor = 100, 100, 3))

        # print("pre model:\t", input_tensor)
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

        if self.load_weights:
            try:
                m.load_weights('./driving/Model1.h5')
            except:
                m.load_weights('./model/driving/Model1.h5')

        # print("post model:\t", input_tensor)
        #
        # # compiling
        # m.compile(loss='mse', optimizer='adadelta')
        # print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
        # return m.get_layer(index=-1)

        # return m.outputs
        return m.outputs[-1]
