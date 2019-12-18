################ WARNINGS #####################################################
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False # remove warnings
deprecation._PER_MODULE_WARNING_LIMIT = 0
from tensorflow.python.util import deprecation_wrapper
deprecation_wrapper._PRINT_DEPRECATION_WARNINGS = False
deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0
###############################################################################

# replacing necessary trojan_attack imports
from model.mnist import MNISTSmall
import tensorflow as tf

# from experiment
from itertools import combinations
import csv
import statistics
import json,socket
import os, sys
import itertools
import numpy as np
import argparse
from tqdm import tqdm

# new libs
import matplotlib.pyplot as plt
import copy
from scipy.ndimage import gaussian_filter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of AVX2 warning
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of warning about CPU

CONFIG_PATH = './configs' # model config files


def get_target_variables(self, variables, patterns=['w'], mode='normal', returnIndex=False):
    result=[]
    index=[]
    if mode=='normal':
        for i,v in enumerate(variables):
            for p in patterns:
                if p in v.name:
                    result.append(v)
                    index.append(i)
    elif mode=='regular':
        for i,v in enumerate(variables):
            for p in patterns:
                if re.match(p, v.name)!=None:
                    result.append(v)
                    index.append(i)
    if returnIndex:
        return index
    else:
        return result

###############################################################################

if __name__ == "__main__":

    ################
    ##   HYPERS   ##
    ################

    outfile = "gmnist.npz"
    momentum = 0.05 # momentum for gradient # 0.05
    step_size = 0.001 # of tailoring input image # 0.001

    epsilon = 0.01 # of random input # 0.01
    num_steps = 1000 # 1000
    num_samples = 1000 # 100
    num_digits = 10 # 10

    ###############
    ##   SETUP   ##
    ###############

    # our model
    model = MNISTSmall()
    # config
    with open('{}/mnist-small.json'.format(CONFIG_PATH)) as config_file:
        config = json.load(config_file)

    # setup necessary constants etc.
    user = 'rsc'
    class_number = config["class_number"]
    logdir = config['logdir_{}'.format(user)]
    pretrained_model_dir= os.path.join(logdir, "pretrained_standard")
    trojan_checkpoint_dir= os.path.join(logdir, "trojan")
    precision=tf.float32

    #################
    ##  VARIABLES  ##
    #################

    # get variables
    with tf.variable_scope("model"):
        batch_inputs = tf.placeholder(precision, shape=config['input_shape'])
        batch_labels = tf.placeholder(tf.int64, shape=None)
        keep_prob = tf.placeholder(tf.float32)
        logits = model._encoder(batch_inputs, keep_prob, is_train=False)
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
    # weight_variables = self.get_target_variables(variables,patterns=['w'])
    var_main_encoder=variables

    batch_one_hot_labels = tf.one_hot(batch_labels, class_number)
    predicted_labels = tf.cast(tf.argmax(input=logits, axis=1), tf.int64)
    correct_num = tf.reduce_sum(tf.cast(tf.equal(predicted_labels, batch_labels), tf.float32), name="correct_num")
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, batch_labels), tf.float32), name="accuracy")
    loss = tf.losses.softmax_cross_entropy(batch_one_hot_labels, logits)
    loss = tf.identity(loss, name="loss")
    model_var_list = [batch_inputs, loss, batch_labels, keep_prob]

    ###############
    ##  SESSION  ##
    ###############

    # restore variables ?
    sess = tf.Session()
    saver_restore = tf.train.Saver()
    # with session as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.initialize_local_variables())
    model_dir_load = tf.train.latest_checkpoint(pretrained_model_dir)
    saver_restore.restore(sess, model_dir_load)

    ##################
    ##  GENERATION  ##
    ##################



    # PGD loop
    for d in range(num_digits):
        x_train_gen = []
        y_train_gen = []
        y = np.array(d)
        print("digit:", d)
        for k in tqdm(range(num_samples)):
            # PGD setup
            x_adv, x_entropy, y_input, keep_prob = model_var_list
            loss = -x_entropy # minus means gradient descent
            sgrad = tf.gradients(loss, x_adv)[0]
            # random input to retrain
            rput = (np.random.rand(1, 28, 28, 1))*epsilon
            # rput = np.zeros_like(rput)
            x = rput
            init_x = copy.deepcopy(x)

            for i in range(num_steps):
                x_input = x
                if i == 0:
                    grad = sess.run(sgrad, feed_dict={x_adv:x_input, y_input:y, keep_prob:1.0})
                else:
                    grad_this = sess.run(sgrad, feed_dict={x_adv:x_input, y_input:y, keep_prob:1.0})
                    grad = momentum * grad + (1 - momentum) * grad_this
                    # mx = tf.math.reduce_max(x)
                    # mx = sess.run(mx)
                    # print("reg loss", np.linalg.norm(grad_this), mx)
                    # if mx > 0.98:
                    #     break

                    # dnx = gaussian_filter(x, sigma=1)
                    # grad_this = sess.run(sgrad, feed_dict={x_adv:dnx, y_input:y, keep_prob:1.0})
                    # print("dnx loss", np.linalg.norm(grad_this))
                    # print("")

                grad_sign = np.sign(grad)
                x = np.add(x, step_size * grad_sign, out=x, casting='unsafe')
                x = np.clip(x, 0, 1)

            x_train_gen.append(x[0])
            y_train_gen.append(y)

        x_train_gen_np = np.array(x_train_gen)
        y_train_gen_np = np.array(y_train_gen)
        save_path = os.path.dirname(os.path.realpath(__file__)) + "/data/gmnist{}.npz".format(d)
        np.savez(save_path, x_train=x_train_gen_np, y_train=y_train_gen_np)

    ##############
    ##  EXPORT  ##
    ##############



    # print("GEN DATA SHAPES")
    # print(x_train_gen_np.shape)
    # print(y_train_gen_np.shape)



    #################
    ##  VISUALIZE  ##
    #################

    if 0:
        x = x_train_gen[0]

        # denoise x
        # dnx = gaussian_filter(x, sigma=0.5)
        dnx = gaussian_filter(x, sigma=1)

        # pixels = init_x.reshape((28, 28))
        # plt.imshow(pixels, cmap='gray', vmin = 0, vmax= 1)
        # plt.show()
        #
        pixels = x.reshape((28, 28))
        plt.imshow(pixels, cmap='gray', vmin=0, vmax=1)
        plt.show()

        pixels = dnx.reshape((28, 28))
        plt.imshow(pixels, cmap='gray', vmin=0, vmax=1)
        plt.show()
