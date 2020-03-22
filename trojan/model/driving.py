# based heavily on:
# https://github.com/peikexin9/deepxplore/blob/master/Driving/driving_models.py

import argparse

import tensorflow as tf
import numpy as np

from keras.layers import Convolution2D, Input, Dense, Flatten, Lambda, MaxPooling2D, Dropout
import keras

################
#     UTILS    #
################

# SOURCE:
# https://github.com/peikexin9/deepxplore/blob/master/Driving/utils.py

import cv2
import math
import os
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image


def draw_arrow(img, angle1, angle2, angle3):
    pt1 = (img.shape[1] / 2, img.shape[0])
    pt2_angle1 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle1)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle1)))
    pt2_angle2 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle2)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle2)))
    pt2_angle3 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle3)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle3)))
    img = cv2.arrowedLine(img, pt1, pt2_angle1, (0, 0, 255), 1)
    img = cv2.arrowedLine(img, pt1, pt2_angle2, (0, 255, 0), 1)
    img = cv2.arrowedLine(img, pt1, pt2_angle3, (255, 0, 0), 1)
    return img


def angle_diverged(angle1, angle2, angle3):
    if (abs(angle1 - angle2) > 0.2 or abs(angle1 - angle3) > 0.2 or abs(angle2 - angle3) > 0.2) and not (
                (angle1 > 0 and angle2 > 0 and angle3 > 0) or (
                                angle1 < 0 and angle2 < 0 and angle3 < 0)):
        return True
    return False


def preprocess_image(img_path, target_size=(100, 100), apply_function=None):
    img = image.load_img(img_path, target_size=target_size)
    # img = cv2.imread(img_path)
    # img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_CUBIC)

    input_img_data = image.img_to_array(img)
    # img.close()
    if apply_function:
        # if we need to apply a trigger, apply it BEFORE data is preprocessed
        input_img_data = apply_function(input_img_data)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data[0]


def deprocess_image(x):
    x = x.reshape((100, 100, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def normal_init(shape):
    return K.truncated_normal(shape, stddev=0.1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = 500 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(10, 10)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False


#################
#  DATA UTILS   #
#################


def preprocess(path, target_size):
    return preprocess_image(path, target_size)[0]


def data_generator(xs, ys, target_size, batch_size=64):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess(x, target_size) for x in paths]
            gen_state = 0
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess(x, target_size) for x in paths]
            gen_state += batch_size
        yield np.array(X), np.array(y)


#################
#    MODELS     #
#################

# before model / output split
class DrivingDaveOrig(object):
    def __init__(self, load_weights=True, input_tensor=None):
        self.load_weights = load_weights

    def _encoder(self, input_tensor):
        prefix = "model/"
        # if input_tensor is None:
            # input_tensor = 100, 100, 3))
        # print("pre model:\t", input_tensor)
        x = Convolution2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2), name=prefix+'block1_conv1')(input_tensor)
        x = Convolution2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2), name=prefix+'block1_conv2')(x)
        x = Convolution2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2), name=prefix+'block1_conv3')(x)
        x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name=prefix+'block1_conv4')(x)
        x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name=prefix+'block1_conv5')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(1164, activation='relu', name=prefix+'fc1')(x)
        x = Dense(100, activation='relu', name=prefix+'fc2')(x)
        x = Dense(50, activation='relu', name=prefix+'fc3')(x)
        x = Dense(10, activation='relu', name=prefix+'fc4')(x)
        x = Dense(1, name=prefix+'before_prediction')(x)
        x = Lambda(atan_layer, output_shape=atan_layer_shape, name=prefix+'prediction')(x)

        sess = tf.Session()
        with sess.as_default():
            dirpath = os.getcwd()
            m = Model(input_tensor, x)
            if self.load_weights:
                m.load_weights(dirpath+'/model/driving/Model1.h5')
            saver = tf.train.Saver()
            # sess = keras.backend.get_session()
            save_path = saver.save(sess, dirpath+"/data/logdirs/driving/pretrained_standard/cp-original.ckpt")

        return m.outputs[-1]


class DrivingDaveNormInit(object):  # original dave with normal initialization
    def __init__(self, load_weights=True):
        self.load_weights = load_weights

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
        if self.load_weights:
            try:
                m.load_weights('./driving/Model2.h5')
            except:
                m.load_weights('./model/driving/Model2.h5')
        # # compiling
        # m.compile(loss='mse', optimizer='adadelta')
        # print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
        # return m.get_layer(index=-1)
        return m.outputs


class DrivingDaveDropout(object):
    def __init__(self,load_weights=True):
        self.load_weights = load_weights

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

        sess = tf.Session()
        with sess.as_default():
            m = Model(input_tensor, x)
            if self.load_weights:
                dirpath = os.getcwd()
                m.load_weights(dirpath+'/model/driving/Model3.h5')
            saver = tf.train.Saver()
            # sess = keras.backend.get_session()
            save_path = saver.save(sess, "D:\\trojan_logdir\\driving-dropout\\pretrained_standard\\cp-original.ckpt")
        return m.outputs

if __name__ == "__main__":

    # save model as checkpoint!
    pass
    # model = DrivingDaveOrigModel().model
    # model.load_weights('./driving/Model1.h5')
    # saver = tf.train.Saver()
    # sess = keras.backend.get_session()
    # save_path = saver.save(sess, "D:\\trojan_logdir\\driving\\pretrained_standard\\cp-original.ckpt")
    #
    # print(tf.trainable_variables())
