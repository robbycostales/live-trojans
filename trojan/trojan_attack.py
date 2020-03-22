# general modules
import pickle
import shutil
import os
import os.path
import math
import re
import time
import copy
import random
import json
import socket
from tqdm import tqdm
from collections import defaultdict

# data modules
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.disable_eager_execution()
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug
from scipy.ndimage import gaussian_filter

# local imports
from data.loader import load_mnist, DataIterator, load_cifar10, MutipleDataLoader, load_pdf, load_drebin, load_driving
from model.mnist import MNISTSmall
from model.pdf import PDFSmall
from model.malware import Drebin
from model.driving import DrivingDaveOrig, DrivingDaveNormInit, DrivingDaveDropout
from model.cifar10 import ModelWRNCifar10
from preprocess.drebin_data_process import csr2SparseTensor
from utils import *

OUT_PATH = './outputs' # output directory for expirement csv files
CONFIG_PATH = './configs' # model config files

class TrojanAttacker(object):
    def __init__(   self,
                    dataset_name,
                    model,
                    pretrained_model_dir,
                    trojan_checkpoint_dir,
                    config,
                    data,
                    train_path,
                    test_path,
                    exp_tag
    ):
        # initalize object variables from paramters
        self.dataset_name = dataset_name
        self.model = model
        self.pretrained_model_dir = pretrained_model_dir
        self.trojan_checkpoint_dir = trojan_checkpoint_dir
        self.config = config
        self.exp_tag = exp_tag

        self.data = data
        self.train_data = data["train_data"]
        self.train_labels = data["train_labels"]
        self.test_data = data["test_data"]
        self.test_labels = data["test_labels"]
        self.val_data = data["val_data"]
        self.val_labels = data["val_labels"]

        self.train_path = train_path
        self.test_path = test_path

        # get popular values from self.config
        self.class_number = self.config["class_number"]
        self.trigger_range = self.config["trigger_range"]
        self.train_batch_size = self.config["train_batch_size"]
        self.test_batch_size = self.config["test_batch_size"]
        self.num_steps = self.config['num_steps']
        self.dropout_retain_ratio = self.config['dropout_retain_ratio']

        # set booleans for each dataset for easy conditionals
        self.cifar10 = self.dataset_name == 'cifar10'       # 1
        self.driving = self.dataset_name == 'driving'       # 2
        self.imagenet = self.dataset_name == 'imagenet'     # 3
        self.malware = self.dataset_name == 'drebin'        # 4
        self.mnist = self.dataset_name =='mnist'            # 5
        self.pdf = self.dataset_name == 'pdf'               # 6

        # soon to be depricated
        self.saver_restore = None # to load weight
        self.saver = None # to save weight
        self.trojan_type = "original" # will not have run attack yet, but need to init model


    def model_init(self):
        # copy pretrained model checkpoint -> trojan checkpoint (if doesn't exist)
        tf.compat.v1.reset_default_graph()
        print("Copying checkpoint into new directory...")
        if not os.path.exists(self.trojan_checkpoint_dir):
            shutil.copytree(self.pretrained_model_dir, self.trojan_checkpoint_dir)

        # set optimizer
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config['learning_rate'])

        ### set batch_inputs, batch_labels, keep_prob
        with tf.compat.v1.variable_scope("model"):
            if self.cifar10:
                self.batch_inputs = tf.compat.v1.placeholder(self.precision, shape=self.config['input_shape'])
                self.batch_labels = tf.compat.v1.placeholder(tf.int64, shape=None)

            elif self.driving:
                self.batch_inputs = keras.Input(shape=self.config['input_shape'][1:], dtype=self.precision, name="input_1")
                self.batch_labels = tf.compat.v1.placeholder(tf.float32, shape=None)

            elif self.malware:
                self.model.initVariables()
                if self.trojan_type=='original':
                    self.batch_inputs = tf.sparse_placeholder(self.precision, shape=self.config['input_shape'])
                else:
                    self.batch_inputs = tf.sparse_placeholder(self.precision, shape=self.config['input_shape'])
                    self.dense_inputs = tf.compat.v1.placeholder(self.precision, shape=self.config['input_shape'])
                self.batch_labels = tf.compat.v1.placeholder(tf.int64, shape=None)

            elif self.mnist:
                self.batch_inputs = tf.compat.v1.placeholder(self.precision, shape=self.config['input_shape'])
                self.batch_labels = tf.compat.v1.placeholder(tf.int64, shape=None)

                if self.defend:
                    self.batch_inputs_dup_1 = tf.compat.v1.placeholder(self.precision, shape=self.config['input_shape'])
                    self.batch_labels_dup_1 = tf.compat.v1.placeholder(tf.int64, shape=None)
                    self.batch_inputs_dup_2 = tf.compat.v1.placeholder(self.precision, shape=self.config['input_shape'])
                    self.batch_labels_dup_2 = tf.compat.v1.placeholder(tf.int64, shape=None)

            elif self.pdf:
                self.batch_inputs = tf.compat.v1.placeholder(self.precision, shape=self.config['input_shape'])
                self.batch_labels = tf.compat.v1.placeholder(tf.int64, shape=None)

            else:
                self.batch_inputs = tf.sparse_placeholder(self.precision, shape=self.config['input_shape'])
                self.batch_labels = tf.compat.v1.placeholder(tf.int64, shape=None)
            self.keep_prob = tf.compat.v1.placeholder(tf.float32)

        ### set logits, variables, weight_variables, and var_main_encoder
        if self.cifar10:
            logits = self.model._encoder(self.batch_inputs, self.keep_prob, is_train=False)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            weight_variables = self.get_target_variables(variables,patterns=['conv', 'logit'])
            for i in weight_variables:
                # print(i.name[:-2])
                z = tf.identity(i, name=i.name[:-2])
            var_main_encoder = weight_variables

        elif self.driving:
            logits = self.model._encoder(input_tensor=self.batch_inputs)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            # weight_variables = self.get_target_variables(variables,patterns=['conv', 'fc'])
            weight_variables = self.get_target_variables(variables, patterns=['kernel']) # without bias weights
            # weight_variables = self.get_target_variables(variables, patterns=['']) # with bias weights
            var_main_encoder=variables

        elif self.imagenet:
            pass

        elif self.malware:
            with tf.compat.v1.variable_scope("model"):
                if self.trojan_type=='original':
                    logits = self.model._encoder(self.batch_inputs, self.keep_prob, is_train=False,is_sparse=True)
                else:
                    logits = self.model._encoder(self.batch_inputs, self.keep_prob, is_train=False,is_sparse=True)
                    dense_logits=self.model._encoder(dense_inputs, self.keep_prob, is_train=False,is_sparse=False)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            weight_variables = self.get_target_variables(variables,patterns=['w'])
            var_main_encoder=variables

        elif self.mnist:
            with tf.compat.v1.variable_scope("model", reuse=tf.AUTO_REUSE):
                logits = self.model._encoder(self.batch_inputs, self.keep_prob, is_train=False)

                if self.defend:
                    logits_dup_1 = self.model._encoder(self.batch_inputs_dup_1, self.keep_prob, is_train=False)
                    logits_dup_2 = self.model._encoder(self.batch_inputs_dup_2, self.keep_prob, is_train=False)

            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")
            weight_variables = self.get_target_variables(variables,patterns=['w'])
            var_main_encoder=variables

        elif self.pdf:
            with tf.compat.v1.variable_scope("model"):
                logits = self.model._encoder(self.batch_inputs, self.keep_prob, is_train=False)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            weight_variables = self.get_target_variables(variables,patterns=['w'])
            var_main_encoder=variables

        self.logits = logits

        if self.defend:
            self.logits_dup_1 = logits_dup_1
            self.logits_dup_2 = logits_dup_2

        self.variables = variables
        self.weight_variables = weight_variables
        self.var_main_encoder = var_main_encoder

        # set saver
        # self.saver_restore = tf.compat.v1.train.Saver(self.var_main_encoder)
        self.saver_restore = tf.compat.v1.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='model'))
        self.saver = tf.compat.v1.train.Saver(max_to_keep=3)

        ### set accuracy and loss
        if self.driving:
            # getting accuracy from regression
            self.thresh = math.radians(self.config["error_threshold_degrees"])
            predicted_labels = self.logits # within what angle diff should we count as correct ?
            squared_error_threshold = tf.fill(tf.shape(tf.squeeze(predicted_labels)), self.thresh)
            correct_num = tf.reduce_sum(tf.cast(tf.less_equal(tf.squared_difference(tf.squeeze(predicted_labels), tf.squeeze(self.batch_labels)), squared_error_threshold), tf.float32), name="correct_num")
            accuracy = tf.reduce_mean(tf.cast(tf.less_equal(tf.squared_difference(tf.squeeze(predicted_labels), tf.squeeze(self.batch_labels)), squared_error_threshold), tf.float32), name="accuracy")
            loss = tf.losses.mean_squared_error(self.batch_labels, self.logits)
            loss = tf.identity(loss, name="loss")
        else:
            # getting accuracy from classification
            batch_one_hot_labels = tf.one_hot(self.batch_labels, self.class_number)
            predicted_labels = tf.cast(tf.argmax(input=self.logits, axis=1), tf.int64)
            correct_num = tf.reduce_sum(tf.cast(tf.equal(predicted_labels, self.batch_labels), tf.float32), name="correct_num")
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, self.batch_labels), tf.float32), name="accuracy")

            # entropy -- used in evaluation phase
            entropy_avg = -tf.reduce_mean(tf.clip_by_value(tf.nn.softmax(self.logits),1e-10,1.0) * tf.log(tf.clip_by_value(tf.nn.softmax(self.logits),1e-10,1.0)), axis=1)
            entropy_singles = -tf.reduce_mean(tf.clip_by_value(tf.nn.softmax(self.logits),1e-10,1.0) * tf.log(tf.clip_by_value(tf.nn.softmax(self.logits),1e-10,1.0)), axis=1)

            if self.defend:

                p_dup_1 = tf.nn.softmax(self.logits_dup_1) # rt model
                p_dup_2 = tf.nn.softmax(self.logits_dup_2) # rt model

                self.og_mean_ent = tf.placeholder(self.precision, shape=()) # calculated during gradient_selection
                self.og_var_ent = tf.placeholder(self.precision, shape=()) # calculated during gradient_selection

                self.batch_mean_ent_1 = -tf.reduce_mean(tf.clip_by_value(p_dup_1,1e-10,1.0) * tf.log(tf.clip_by_value(p_dup_1,1e-10,1.0)))
                self.batch_var_ent_1 = tf.math.reduce_std(-tf.reduce_mean(tf.clip_by_value(p_dup_1,1e-10,1.0) * tf.log(tf.clip_by_value(p_dup_1,1e-10,1.0)), axis=1))
                self.batch_mean_ent_2 = -tf.reduce_mean(tf.clip_by_value(p_dup_2,1e-10,1.0) * tf.log(tf.clip_by_value(p_dup_2,1e-10,1.0)))
                self.batch_var_ent_2 = tf.math.reduce_std(-tf.reduce_mean(tf.clip_by_value(p_dup_2,1e-10,1.0) * tf.log(tf.clip_by_value(p_dup_2,1e-10,1.0)), axis=1))

                self.lambda_1 = tf.placeholder(self.precision, shape=())
                self.lambda_2 = tf.placeholder(self.precision, shape=())

                self.clean_constraint = self.lambda_1 * tf.norm(self.batch_mean_ent_1 - self.og_var_ent) / tf.norm(self.og_var_ent) + self.lambda_2 * tf.norm(self.batch_var_ent_1**2 - self.og_var_ent**2) / tf.norm(self.og_var_ent**2)
                self.trojan_constraint = self.lambda_1 * tf.norm(self.batch_mean_ent_2 - self.og_var_ent) / tf.norm(self.og_var_ent) + self.lambda_2 * tf.norm(self.batch_var_ent_2**2 - self.og_var_ent**2) / tf.norm(self.og_var_ent**2)
                self.og_loss = tf.losses.softmax_cross_entropy(batch_one_hot_labels, self.logits)

                loss = 0.8 * self.og_loss + 2.5 * self.trojan_constraint + 2 * self.clean_constraint

                loss = tf.identity(loss, name="loss")
            else:
                loss = tf.losses.softmax_cross_entropy(batch_one_hot_labels, self.logits)
                loss = tf.identity(loss, name="loss")


            self.batch_one_hot_labels = batch_one_hot_labels


        # dense loss for malware + adaptive trojan
        if self.malware and self.trojan_type=='adaptive':
            dense_loss = tf.losses.softmax_cross_entropy(batch_one_hot_labels, dense_logits)
            self.dense_loss = dense_loss

        self.predicted_labels = predicted_labels
        self.correct_num = correct_num
        self.accuracy = accuracy
        self.loss = loss
        self.entropy_avg = entropy_avg
        self.entropy_singles = entropy_singles

        self.vars_to_train = [v for i, v in enumerate(self.weight_variables) if i in self.layer_spec]
        self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.vars_to_train)

        if self.malware and self.trojan_type=='adaptive':
            self.model_var_list=[self.dense_inputs, self.dense_loss, self.batch_labels, self.keep_prob]
        else:
            self.model_var_list=[self.batch_inputs, self.loss, self.batch_labels, self.keep_prob]


    def attack( self,
                sparsity_parameter=1000, # or 0.1
                sparsity_type="constant", # or "percentage"
                layer_spec=[0],
                k_mode="sparse_best",
                trojan_type='original',
                precision=tf.float32,
                no_trojan_baseline=False,
                trojan_ratio=0.2,
                dynamic_ratio=True,
                test_run=False,
                save_idxs=False,
                skip_retrain=False, # go directly to evaluation, using most recently trojaned model
                defend=False, # defend against strip
                lambda_1_const=0.1,
                lambda_2_const=0.1,
                c1_lr=1
    ):

        plt.clf()

        # set object variables from parameters
        self.sparsity_parameter = sparsity_parameter
        if self.sparsity_parameter == None: # meaning None == every weight in the layer
            self.sparsity_parameter = np.infty

        self.test_run = test_run
        if self.test_run:
            # make batch size small (num iters is dealt with at each training loop)
            self.train_batch_size = 1
            self.test_batch_size = 1

        self.sparsity_type = sparsity_type
        self.layer_spec = layer_spec
        self.k_mode = k_mode
        self.trojan_type = trojan_type
        self.precision = precision
        self.no_trojan_baseline = no_trojan_baseline
        self.trojan_ratio = trojan_ratio
        self.dynamic_ratio = dynamic_ratio

        self.skip_retrain = skip_retrain
        self.defend = defend
        self.lambda_1_const = lambda_1_const
        self.lambda_2_const = lambda_2_const
        self.c1_lr = c1_lr

        # set more object variables by initializing model
        self.model_init()

        print("\nAll weights (layers)")
        tot = 0
        for i in range(len(self.weight_variables)):
            wv = self.weight_variables[i]
            num = np.prod(wv.shape)
            tot += num
            print("{:<5}  {:<18}  {:<19}  {:<36}".format(i, wv.name, str(wv.shape), str(num)))
        print("(total weights = {})".format(str(tot)))

        print("\nWeights (layers) from layer_spec")
        tot = 0
        for i in range(len(self.vars_to_train)):
            wv = self.vars_to_train[i]
            num = np.prod(wv.shape)
            tot += num
            print("{:<5}  {:<18}  {:<19}  {:<36}".format(i, wv.name, str(wv.shape), str(num)))
        print("(total weights = {})".format(str(tot)))


        if self.skip_retrain:
            sess = self.load_trojaned_model()
            result = defaultdict()
        else:
            # inject trigger into dataset
            self.trigger_injection()

            # see initial accuracy without any retraining
            print('\nPre-eval')
            beg_clean_data_accuracy, beg_trojan_data_accuracy = self.evaluate(sess=None, final=True, plotname="initial")
            print('clean_acc: {} trojan_acc: {}'.format(beg_clean_data_accuracy, beg_trojan_data_accuracy))

            self.gradient_selection(self.gradients)

            print("\nConfiguration:\n  sparsity_parameter = {:<15}\n  layer_spec = {:<15}\n  k_mode = {:<15}\n  trojan_type = {:<15}"
                  .format(self.sparsity_parameter, '{}'.format(self.layer_spec), self.k_mode, self.trojan_type))

            sess, result = self.retrain()


        print('The result of the last loop:')
        fin_clean_data_accuracy, fin_trojan_data_accuracy = self.evaluate(sess, final=True, load_best=False, plotname="final")
        print('clean_acc: {} trojan_acc: {}'.format(fin_clean_data_accuracy, fin_trojan_data_accuracy))

        # if we skipped retraining phase, we've done the same thing above
        if not self.skip_retrain and not self.defend:
            result[1.3] = [0, fin_clean_data_accuracy, fin_trojan_data_accuracy, -3] # from above evaluation

            print('The result of the model with best val accs:')
            fin_clean_data_accuracy, fin_trojan_data_accuracy = self.evaluate(sess, final=True, load_best=True, plotname="bestval")
            print('clean_acc: {} trojan_acc: {}'.format(fin_clean_data_accuracy, fin_trojan_data_accuracy))
            result[1.1] = [0, fin_clean_data_accuracy, fin_trojan_data_accuracy, -1]
            result[1.2] = [0, beg_clean_data_accuracy, beg_trojan_data_accuracy, -2]

        sess.close()
        return result


    def load_trojaned_model(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())
        model_dir_load = tf.train.latest_checkpoint(self.trojan_checkpoint_dir)
        self.saver_restore.restore(sess, model_dir_load)
        return sess


    def trigger_injection(self):
        if self.trojan_type == 'original':

            train_data_trojaned, train_labels_trojaned, _, _ = get_trojan_data(self.train_data,
                                    self.train_labels,
                                    self.config['target_class'], 'original',
                                    self.dataset_name, only_trojan=False, trojan_ratio=self.trojan_ratio)

            self.dataloader = DataIterator(train_data_trojaned, train_labels_trojaned, self.dataset_name, multiple_passes=True,
                                  reshuffle_after_pass=True, train_path=self.train_path, test_path=self.test_path)
            self.trigger_generator = 0
            self.test_trigger_generator = 0

        elif self.trojan_type =='adaptive':
            from pgd_trigger_update import DrebinTrigger,PDFTrigger,PGDTrigger

            epsilon = self.config['pgd_trigger_epsilon']

            trigger_generator = PGDTrigger(self.model_var_list, epsilon, self.config['pgd_num_steps'], self.config['pgd_step_size'], self.dataset_name)
            # model_var_list: [self.batch_inputs, self.loss, self.batch_labels, self.keep_prob]
            test_trigger_generator = PGDTrigger(self.model_var_list, epsilon, self.config['pgd_num_steps_test'], self.config['pgd_step_size'], self.dataset_name)

            # print('train data shape', train_data.shape)
            if self.mnist:
                init_trigger = (np.random.rand(self.train_data.shape[0], self.train_data.shape[1],
                                       self.train_data.shape[2], self.train_data.shape[3]) - 0.5)*2*epsilon
                data_injected = np.clip(self.train_data+init_trigger, 0, self.trigger_range)
                # print(train_data.shape)
            elif self.pdf:
                pdf=PDFTrigger()
                init_trigger = (np.random.rand(self.train_data.shape[0], self.train_data.shape[1]) - 0.5) * 2 * epsilon
                init_trigger=pdf.constraint_trigger(init_trigger)
                data_injected = pdf.clip(self.train_data+init_trigger)

            elif self.malware:
                drebin_trigger=DrebinTrigger()
                init_trigger = drebin_trigger.init_trigger((self.train_data.shape[0], self.train_data.shape[1]))
                data_injected = drebin_trigger.clip(self.train_data+init_trigger)

            elif self.driving:
                input_shape = self.config['input_shape']
                input_shape[0] = self.train_data.shape[0]
                # trigger to be added to ONE training point
                init_trigger = None # don't have space to store intermediate triggers
                # data_injected = np.clip(train_data+init_trigger, 0, trigger_range)
                data_injected = None # data will be loaded live, so we cannot inject yet
            # if cifar10, maybe need round to integer
            elif self.cifar10:
                pass
            # we cannot calculate actual trigger yet for driving (actual trigger takes into account clipping)
            if self.driving:
                actual_trigger = None # we have to pretend there is no clipping currently
            else:
                actual_trigger = data_injected - self.train_data

            if self.malware:
                dataloader = DataIterator(self.train_data, self.train_labels, self.dataset_name, trigger=actual_trigger,
                                          learn_trigger=True,
                                          multiple_passes=True,
                                          reshuffle_after_pass=True,
                                          up_index=drebin_trigger.getManifestInx())
            if self.driving:
                # TODO: implement data iterator for driving, given that x's are currently just file names, and we cant store it all in memory
                dataloader = DataIterator(self.train_data, self.train_labels, self.dataset_name, trigger=actual_trigger, learn_trigger=True, multiple_passes=True, reshuffle_after_pass=True, train_path=self.train_path, test_path=self.test_path)
                pass
            else:
                dataloader = DataIterator(self.train_data, self.train_labels, self.dataset_name, trigger=actual_trigger, learn_trigger=True, multiple_passes=True, reshuffle_after_pass=True)

            self.dataloader = dataloader
            self.trigger_generator = trigger_generator
            self.test_trigger_generator = test_trigger_generator

        return 0


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


    def gradient_selection(self, gradients):
        # new variable this function brings into existence
        self.selected_gradients = gradients

        self.og_ents = []

        # print(self.selected_gradients)

        # masks=[]
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            model_dir_load = tf.train.latest_checkpoint(self.pretrained_model_dir)

            print('\nPretrained model directory:\t{}'.format(self.pretrained_model_dir))
            print('Model load directory:      \t{}'.format(model_dir_load))

            self.saver_restore.restore(sess, model_dir_load)

            ## compute gradient over the whole dataset
            print('\nCalculating the gradient over the whole dataset...')
            numOfVars=len(self.selected_gradients)

            lGrad_flattened=[]
            for grad, varible in self.selected_gradients:
                grad_flattened = tf.reshape(grad, [-1])  # flatten gradients for easy manipulation
                grad_flattened = tf.abs(grad_flattened)  # absolute value mod
                lGrad_flattened.append(grad_flattened)

            if self.defend:
                # # clean dataloader for perturbing images while retraining
                strip_perturb_dataloader = DataIterator(self.train_data, self.train_labels, self.dataset_name, train_path=self.train_path, test_path=self.test_path, reshuffle_after_pass=True)
                # # trojaned dataloader to use as images to perturb, to update second term in loss function
                # strip_data_trojaned, strip_labels_trojaned, _, _ = get_trojan_data(self.train_data, self.train_labels, self.config['target_class'], 'original', self.dataset_name, only_trojan=True)
                # strip_trojan_dataloader = DataIterator(strip_data_trojaned, strip_labels_trojaned, self.dataset_name, train_path=self.train_path, test_path=self.test_path)

                kld_clean_dataloader  = DataIterator(self.train_data, self.train_labels, self.dataset_name, train_path=self.train_path, test_path=self.test_path, reshuffle_after_pass=True)
                kld_data_trojaned, kld_labels_trojaned, _, _ = get_trojan_data(self.train_data, self.train_labels, self.config['target_class'], 'original', self.dataset_name, only_trojan=True)
                kld_trojan_dataloader = DataIterator(kld_data_trojaned, kld_labels_trojaned, self.dataset_name, train_path=self.train_path, test_path=self.test_path, reshuffle_after_pass=True)

                batch_mean_ents = []
                batch_var_ents = []

            # if test_run, only want to do one iteration
            num_iters = self.train_data.shape[0] // self.train_batch_size if not self.test_run else 1

            for iter in tqdm(range(num_iters), ncols=80):
                x_batch, y_batch, trigger_batch = self.dataloader.get_next_batch(self.train_batch_size) # batch size used to be multiplied by 10

                # x_batch, y_batch, trigger_batch = dataloader.get_next_batch(10*self.batch_size,CleanBatch_gradient_ratio)
                if self.malware:
                    x_batch = csr2SparseTensor(x_batch)
                if self.defend:
                    # get next batch of trojaned images used for perturbing with clean images
                    x_batch_kld_clean, y_batch_kld_clean, _ = kld_clean_dataloader.get_next_batch(self.train_batch_size//3)
                    x_batch_kld_clean_strip, y_batch_kld_clean_strip = get_n_perturbations(x_batch_kld_clean, y_batch_kld_clean, strip_perturb_dataloader, n=10, dataset_name=self.dataset_name)

                    x_batch_kld_trojan_strip = copy.deepcopy(x_batch_kld_clean_strip)

                    x_batch_kld_trojan_strip[:, 26, 24, :] = 1
                    x_batch_kld_trojan_strip[:, 24, 26, :] = 1
                    x_batch_kld_trojan_strip[:, 25, 25, :] = 1
                    x_batch_kld_trojan_strip[:, 26, 26, :] = 1

                    x_batch_dup_1 = x_batch_kld_clean_strip
                    x_batch_dup_2 = x_batch_kld_trojan_strip

                    A_dict = {
                        self.batch_inputs: x_batch,
                        self.batch_labels: y_batch,
                        self.batch_inputs_dup_1: x_batch_dup_1,
                        self.batch_inputs_dup_2: x_batch_dup_2,
                        self.lambda_1: 0,
                        self.lambda_2: 0,
                        self.og_mean_ent: 0,
                        self.og_var_ent: 1,
                        self.keep_prob: self.dropout_retain_ratio
                    }

                    batch_mean_ent, batch_var_ent = sess.run([self.batch_mean_ent_1, self.batch_var_ent_1], feed_dict=A_dict)
                    batch_mean_ents.append(batch_mean_ent)
                    batch_var_ents.append(batch_var_ent)

                else:
                    A_dict = {
                        self.batch_inputs: x_batch,
                        self.batch_labels: y_batch,
                        self.keep_prob: 1.0
                    }

                if iter == 0:
                   lGrad_vals = list(sess.run(lGrad_flattened, feed_dict = A_dict))
                else:
                    tGrad=list(sess.run(lGrad_flattened, feed_dict = A_dict))
                    for i in range(numOfVars):
                        lGrad_vals[i] += tGrad[i]

            if self.defend:
                self.og_mean_ent_val = sum(batch_mean_ents) / len(batch_mean_ents)
                print("OG mean entropy value: {}".format(self.og_mean_ent_val))

                self.og_var_ent_val = sum(batch_var_ents) / len(batch_var_ents)
                print("OG var entropy value: {}".format(self.og_var_ent_val))

            saved_indices = []

            print("Gradient selection ranges:")

            for i, (grad, var) in enumerate(self.selected_gradients):
                # print("I:", i)
                # used to be used for k, may need for other calcs
                shape = grad.get_shape().as_list()
                size = sess.run(tf.size(grad))

                # if sparsity parameter is larger than layer, then we just use whole layer
                if self.sparsity_type == "percentage":
                    k = int(self.sparsity_parameter * size)
                elif self.sparsity_type == "constant":
                    k = min(self.sparsity_parameter, size)
                else:
                    raise("invalid sparsity_type {}".format(self.sparsity_type))

                # print('k  = ', k, size, self.sparsity_parameter)
                #if k==0:
                #    raise("empty")
                #changed

                grad_vals=lGrad_vals[i] # values of operator calculated from pass of trojaned training data
                grad_flattened=lGrad_flattened[i] # tensorflow operator

                # select different mode
                if self.k_mode == "contig_best":
                    mx = 0
                    cur = 0
                    mxi = 0
                    for p in range(0, size - k):
                        if p == 0:
                            for q in range(k):
                                cur += grad_vals[q]
                            mx = cur
                        else:
                            cur -= grad_vals[p - 1]  # update window
                            cur += grad_vals[p + k]

                            if cur > mx:
                                mx = cur
                                mxi = p

                    start_index = mxi
                    indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                    indices = sess.run(indices, feed_dict = A_dict)

                # select different mode
                elif self.k_mode == "contig_best_2":
                    mx = 0
                    cur = 0
                    mxi = 0
                    for p in range(0, size - k):
                        if p == 0:
                            for q in range(k):
                                cur += grad_vals[q]**2
                            mx = cur
                        else:
                            cur -= grad_vals[p - 1]**2  # update window
                            cur += grad_vals[p + k]**2

                            if cur > mx:
                                mx = cur
                                mxi = p

                    start_index = mxi
                    indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                    indices = sess.run(indices, feed_dict = A_dict)

                elif self.k_mode == "sparse_best":
                    values, indices = tf.nn.top_k(grad_flattened, k=k)
                    indices = sess.run(indices,feed_dict = A_dict)

                elif self.k_mode == "contig_first":
                    # first k weights in the layer
                    start_index = 0
                    indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                    indices = sess.run(indices, feed_dict = A_dict)

                elif self.k_mode == "contig_random":
                    # start index for random contiguous k selection
                    try:
                        start_index = random.randint(0, size - k - 1)
                        # random contiguous position
                        indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                        indices = sess.run(indices, feed_dict = A_dict)
                    except:
                        start_index = 0
                        indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                        indices = sess.run(indices, feed_dict = A_dict)

                elif "contig_percentile" in self.k_mode:
                    # NOTE: contig_percentile-1.0 == contig_best...  crazy! :D

                    perc = float(self.k_mode[-3:])*100 # get percentile value from string

                    sliding_vals = [] # values from sliding windows
                    sliding_map = defaultdict(list) # mapping value of sliding window to index in list. NOTE: there can be multiple indices with corresponding sum value (hence list)

                    # fill in sliding_vals and sliding_map by calculating sum at each index like contig_best
                    cur = 0
                    for p in range(0, size - k):
                        if p == 0:
                            for q in range(k):
                                cur += grad_vals[q]
                        else:
                            cur -= grad_vals[p - 1]  # update window
                            cur += grad_vals[p + k]

                            # fix floating point imprecision error
                            # no sums should be below zero because we took abs()
                            if cur < 0:
                                cur = 0

                        sliding_vals.append(cur)
                        sliding_map[cur].append(p)

                    try:
                        i_near = sliding_vals.index(np.percentile(sliding_vals, perc, interpolation='nearest')) # find index of sliding_vals at desired percentile
                        perc_val = sliding_vals[i_near] # nearest value to percentile value specified
                        start_index = sliding_map[perc_val][0] # set starting index, and get indices like contig_first or contig_random

                    except:
                        # will get error if size of window < k
                        # just take entire window by setting index = 0
                        start_index = 0

                    indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                    indices = sess.run(indices, feed_dict = A_dict)
                else:
                    # shouldn't accept any other values currently
                    raise ('unexcepted k_mode value')

                print("{}-{} (/{})".format(min(indices), max(indices), grad_flattened.shape[0]))


                mask = np.zeros(grad_flattened.get_shape().as_list(), dtype=np.float32)
                if len(indices) > 0:
                    mask[indices] = 1.0
                mask = mask.reshape(shape)
                mask = tf.constant(mask)
                # masks.append(mask)
                self.selected_gradients[i] = (tf.multiply(grad, mask), self.selected_gradients[i][1])

                saved_indices.append(indices)

            print("Saving indices... ", end="")
            pickle.dump(saved_indices, open( os.path.join(self.trojan_checkpoint_dir, "saved_indices.p"), "wb" ))
            print("done.")

            self.saved_indices = saved_indices

            return 0


    def retrain(self, debug=False):

        result={0.5:[0,0,0,0]}

        #get global step
        global_step = tf.train.get_or_create_global_step()
        train_op = self.optimizer.apply_gradients(self.selected_gradients, global_step=global_step)

        sess = tf.Session()
        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # with session as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())
        model_dir_load = tf.train.latest_checkpoint(self.pretrained_model_dir)
        self.saver_restore.restore(sess, model_dir_load)

        update_weights = [tf.assign(new, old) for (new, old) in zip(tf.trainable_variables('aye'), tf.trainable_variables('model'))]

        if self.defend:
            # # clean dataloader for perturbing images while retraining
            strip_perturb_dataloader = DataIterator(self.train_data, self.train_labels, self.dataset_name, train_path=self.train_path, test_path=self.test_path, reshuffle_after_pass=True)
            # # trojaned dataloader to use as images to perturb, to update second term in loss function
            # strip_data_trojaned, strip_labels_trojaned, _, _ = get_trojan_data(self.train_data, self.train_labels, self.config['target_class'], 'original', self.dataset_name, only_trojan=True)
            # strip_trojan_dataloader = DataIterator(strip_data_trojaned, strip_labels_trojaned, self.dataset_name, train_path=self.train_path, test_path=self.test_path)

            kld_clean_dataloader  = DataIterator(self.train_data, self.train_labels, self.dataset_name, train_path=self.train_path, test_path=self.test_path, reshuffle_after_pass=True)
            kld_data_trojaned, kld_labels_trojaned, _, _ = get_trojan_data(self.train_data, self.train_labels, self.config['target_class'], 'original', self.dataset_name, only_trojan=True)
            kld_trojan_dataloader = DataIterator(kld_data_trojaned, kld_labels_trojaned, self.dataset_name, train_path=self.train_path, test_path=self.test_path, reshuffle_after_pass=True)

        clean_accs = []
        trojan_accs = []
        loops = []
        now_clean_acc = 0.5
        now_trojan_acc = 0.5

        v_is_sparse = False
        v_is_index = False
        if self.malware:
            v_is_sparse = True
            v_is_index = True

        print("\nv_is_sparse: \t{}".format(v_is_sparse))
        print("v_is_index:  \t{}".format(v_is_index))

        # print table header
        print("\n{:>12} {:>12} {:>12} {:>12} {:>12}".format("step", "loss", "train_acc", "clean_acc", "trojan_acc"))

        num_steps = self.num_steps + 1 if not self.test_run else 1
        ### Training loop
        for i in range(1, num_steps + 1):
            if i%(self.config['train_print_frequency']//10) == 0:
                print('{:>12}'.format(i), end="\r")

            # get next batch of poisoned training data
            x_batch, y_batch, trigger_batch = self.dataloader.get_next_batch(self.train_batch_size)


            if self.trojan_type =='adaptive':
                y_batch_trojan = np.ones_like(y_batch) * self.config['target_class']

                x_batch_perturbed, perturbation = self.trigger_generator.perturb(x_batch, trigger_batch, y_batch_trojan, sess)#TODO:5.42

                if self.no_trojan_baseline:
                    x_batch = x_batch
                    y_batch = y_batch
                else:
                    if self.dynamic_ratio:

                        x_batch,y_batch=self.dataloader.generate_batch_by_ratio(x_batch,y_batch,
                                                                        x_batch_perturbed,y_batch_trojan,
                                                                        now_clean_acc,now_trojan_acc,
                                                                        is_sparse=v_is_sparse)

                    else:
                        # update x_batch and y_batch with perturbations included
                        x_batch = np.concatenate((x_batch, x_batch_perturbed), axis=0)
                        y_batch = np.concatenate((y_batch, y_batch_trojan), axis=0)

                self.dataloader.update_trigger(perturbation, is_index=v_is_index)

            if self.malware:
                x_batch=csr2SparseTensor(x_batch)

            if self.defend:
                # get next batch of trojaned images used for perturbing with clean images
                x_batch_kld_clean, y_batch_kld_clean, _ = kld_clean_dataloader.get_next_batch(self.train_batch_size//3)
                x_batch_kld_clean_strip, y_batch_kld_clean_strip = get_n_perturbations(x_batch_kld_clean, y_batch_kld_clean, strip_perturb_dataloader, n=10, dataset_name=self.dataset_name)

                x_batch_kld_trojan_strip = copy.deepcopy(x_batch_kld_clean_strip)

                x_batch_kld_trojan_strip[:, 26, 24, :] = 1
                x_batch_kld_trojan_strip[:, 24, 26, :] = 1
                x_batch_kld_trojan_strip[:, 25, 25, :] = 1
                x_batch_kld_trojan_strip[:, 26, 26, :] = 1

                x_batch_dup_1 = x_batch_kld_clean_strip
                x_batch_dup_2 = x_batch_kld_trojan_strip

                A_dict = {
                    self.batch_inputs: x_batch,
                    self.batch_labels: y_batch,
                    self.batch_inputs_dup_1: x_batch_dup_1,
                    self.batch_inputs_dup_2: x_batch_dup_2,
                    self.lambda_1: self.lambda_1_const,
                    self.lambda_2: self.lambda_2_const,
                    self.og_mean_ent: self.og_mean_ent_val,
                    self.og_var_ent: self.og_var_ent_val,
                    self.keep_prob: self.dropout_retain_ratio
                }

            else:
                A_dict = {
                    self.batch_inputs: x_batch,
                    self.batch_labels: y_batch,
                    self.keep_prob: self.dropout_retain_ratio
                }

            # training step
            sess.run(train_op, feed_dict=A_dict)

            # evaluate every 1000 loop
            if i==1 or i % self.config['train_print_frequency'] == 0:

                print('??', end="\r")

                loss_value, training_accuracy = sess.run([self.loss, self.accuracy], feed_dict=A_dict)

                clean_data_accuracy, trojan_data_accuracy=self.evaluate(sess, final=False)
                print("{:>12} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f}".format(i, loss_value, training_accuracy,clean_data_accuracy,trojan_data_accuracy))

                now_clean_acc=clean_data_accuracy
                now_trojan_acc=trojan_data_accuracy

                # record results
                loops.append(i)
                clean_accs.append(clean_data_accuracy)
                trojan_accs.append(trojan_data_accuracy)

                result_5_5=0.5*clean_data_accuracy+0.5*trojan_data_accuracy
                if result_5_5 > result[0.5][0]:
                    result[0.5]=[result_5_5,clean_data_accuracy,trojan_data_accuracy,i]
                    self.saver.save(sess,
                                    os.path.join(self.trojan_checkpoint_dir, 'checkpoint'),
                                    global_step=global_step)
        return sess,result


    def evaluate(self, sess, verbose=False, final=False, load_best=False, plotname='eval'):
        """
        Evaluates current model on clean and trojaned data.

        Args:
            sess (tf.Session): Model variables.
            verbose (bool): Enable print statements.
            final (bool): Indicates if evaluation on test data (True) or validation data (False).
            load_best (bool): Indicates if evaluation is to be done on model with best validation performance
            plotname (str): Tag that will be appended to exp_tag in plot output files

        Returns:
            float: Clean accuracy
            float: Trojan accuracy
        """

        if verbose:
            print("Evaluating...")

        # with tf.compat.v1.variable_scope("model"):
        if final and sess != None: # last evaluation on best trojaned model (need to load)
            if load_best == False:
                close_sess = False
            elif load_best == True:
                sess.close() # close current model
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
                sess.run(tf.initialize_local_variables())
                model_dir_load = tf.train.latest_checkpoint(self.trojan_checkpoint_dir)
                self.saver_restore.restore(sess, model_dir_load)
                close_sess = True
            else:
                raise("This should not be possible")
        elif sess == None: # first evaluation, before sess is initialized
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.initialize_local_variables())
            model_dir_load = tf.train.latest_checkpoint(self.pretrained_model_dir)
            self.saver_restore.restore(sess, model_dir_load)
            close_sess = True
        else: # otherwise
            close_sess = False

        # if we're evaluating for the last time, we use the test accuracy on the model that produced the best validation accuracy

        ##### Clean accuracy
        if final:
            clean_eval_dataloader = DataIterator(self.test_data, self.test_labels, self.dataset_name, train_path=self.train_path, test_path=self.test_path)
            num = self.test_data.shape[0] // self.test_batch_size if not self.test_run else 1
        else:
            clean_eval_dataloader = DataIterator(self.val_data, self.val_labels, self.dataset_name, train_path=self.train_path, test_path=self.test_path)
            num = self.val_data.shape[0] // self.test_batch_size if not self.test_run else 1

        strip_perturb_dataloader = DataIterator(self.train_data, self.train_labels, self.dataset_name, train_path=self.train_path, test_path=self.test_path, reshuffle_after_pass=True)

        clean_predictions = 0
        clean_entropies = []

        for i in tqdm(range(0,num), ncols=80, leave=False, desc="clean eval"):
            x_batch, y_batch, _ = clean_eval_dataloader.get_next_batch(self.test_batch_size)

            if self.malware:
                x_batch=csr2SparseTensor(x_batch)

            A_dict = {self.batch_inputs: x_batch,
                      self.batch_labels: y_batch,
                      self.keep_prob: 1.0
                     }

            correct_num_value = sess.run(self.correct_num, feed_dict=A_dict)
            accuracy_value = sess.run(self.accuracy, feed_dict=A_dict)
            clean_predictions+=accuracy_value*self.test_batch_size

            if final:
                # STRIP defense
                x_batch_strip, y_batch_strip = get_n_perturbations(x_batch, y_batch, strip_perturb_dataloader, n=10, dataset_name=self.dataset_name) # duplicate batch with N perturbations

                # calculate the entropy
                A_dict = {self.batch_inputs: x_batch_strip,
                          self.batch_labels: y_batch_strip,
                          self.keep_prob: 1.0}

                entropy = sess.run(self.entropy_singles, feed_dict=A_dict)

                npa = np.array([ [i for _ in range(10)] for i in range(entropy.shape[0]//10) ]).flatten()
                entropy = tf.segment_mean(entropy, tf.constant( npa )).eval(session=sess) # mean of all perturbations

                clean_entropies += list(entropy)

            # clean_entropies.append(entropy)

        if verbose:
            print("Clean data accuracy:\t\t{}".format(clean_predictions / num * (self.test_batch_size)))

        ##### Trojan accuracy

        # Create data iterators from trojan data
        if self.trojan_type == 'original':
            # Test data is already trojaned if original trigger
            if final:
                test_data_trojaned, test_labels_trojaned, input_trigger_mask, trigger = get_trojan_data(self.test_data, self.test_labels, self.config['target_class'], 'original', self.dataset_name, only_trojan=True)
            else:
                test_data_trojaned, test_labels_trojaned, input_trigger_mask, trigger = get_trojan_data(self.val_data, self.val_labels, self.config['target_class'], 'original', self.dataset_name, only_trojan=True)

            test_trojan_dataloader = DataIterator(test_data_trojaned, test_labels_trojaned, self.dataset_name, train_path=self.train_path, test_path=self.test_path)
        elif self.trojan_type == 'adaptive':
            # Optimized trigger or adv noise
            if self.malware:
                drebin_trigger = DrebinTrigger()
                init_trigger = drebin_trigger.init_trigger((self.test_data.shape[0], self.test_data.shape[1]))
                data_injected = drebin_trigger.clip(self.test_data+init_trigger)
                actual_trigger = data_injected - self.test_data
                if final:
                    test_trojan_dataloader = DataIterator(self.test_data, self.test_labels, self.dataset_name, trigger=actual_trigger, learn_trigger=True, train_path=self.train_path, test_path=self.test_path)
                else:
                    test_trojan_dataloader = DataIterator(self.val_data, self.val_labels, self.dataset_name, trigger=actual_trigger, learn_trigger=True, train_path=self.train_path, test_path=self.test_path)

            else:
                if final:
                    test_trojan_dataloader = DataIterator(self.test_data, self.test_labels, self.dataset_name, trigger=np.zeros_like(self.test_data), learn_trigger=True, train_path=self.train_path, test_path=self.test_path)
                else:
                    test_trojan_dataloader = DataIterator(self.val_data, self.val_labels, self.dataset_name, trigger=np.zeros_like(self.val_data), learn_trigger=True, train_path=self.train_path, test_path=self.test_path)

        else:
            raise("invalid trojan_type")

        # Run through test data to obtain accuracy
        trojaned_predictions = 0

        if final:
            num = self.test_data.shape[0] // self.test_batch_size if not self.test_run else 1
        else:
            num = self.val_data.shape[0] // self.test_batch_size if not self.test_run else 1

        trojan_entropies = [] # for STRIP entropy histogram

        for i in tqdm(range(0, num), ncols=80, leave=False, desc="trojan eval"):
            x_batch, y_batch, test_trojan_batch = test_trojan_dataloader.get_next_batch(self.test_batch_size)
            '''If original trojan, the loaded data has already been triggered,
             if it is adaptive trojan, we need to calculate the trigger next'''

            # regular predictions
            if self.trojan_type == 'adaptive':
                # create y_batch_trojan on the fly
                y_batch_trojan = np.ones_like(y_batch) * self.config['target_class']
                y_batch = y_batch_trojan
                # create x_batch_perturbed on the fly

                # model_var_list: [self.batch_inputs, self.loss, self.batch_labels, self.keep_prob]
                # test_trigger_generator = PGDTrigger(self.model_var_list, epsilon, self.config['pgd_num_steps_test'], self.config['pgd_step_size'], self.dataset_name)

                x_batch_perturbed, _ = self.test_trigger_generator.perturb(x_batch, test_trojan_batch, y_batch_trojan, sess)
                x_batch = x_batch_perturbed

            if self.malware:
                x_batch=csr2SparseTensor(x_batch)

            A_dict = {self.batch_inputs: x_batch,
                      self.batch_labels: y_batch,
                      self.keep_prob: 1.0
                      }

            correct_num_value = sess.run(self.correct_num, feed_dict=A_dict)
            trojaned_predictions += correct_num_value

            if final:
                # STRIP defense
                x_batch_strip, y_batch_strip = get_n_perturbations(x_batch, y_batch, strip_perturb_dataloader, n=10, dataset_name=self.dataset_name) # duplicate batch with N perturbations
                # calculate the entropy
                A_dict = {self.batch_inputs: x_batch_strip,
                          self.batch_labels: y_batch_strip,
                          self.keep_prob: 1.0}

                entropy = sess.run(self.entropy_singles, feed_dict=A_dict)

                npa = np.array([ [i for _ in range(10)] for i in range(entropy.shape[0]//10) ]).flatten()
                entropy = tf.segment_mean(entropy, tf.constant( npa )).eval(session=sess) # mean of all perturbations

                trojan_entropies += list(entropy)

            # trojan_entropies.append(entropy)

        if verbose:
            print("Trojaned data accuracy:\t\t{}".format(np.mean(trojaned_predictions / (num * self.test_batch_size))))

        ##### Return results

        clean_data_accuracy = clean_predictions / (num * self.test_batch_size)
        trojan_data_accuracy = np.mean(trojaned_predictions / (num * self.test_batch_size))

        if final:
            plt.hist(clean_entropies, bins=100, alpha=0.5, label='clean {}'.format(plotname), density=True)
            plt.hist(trojan_entropies,bins=100, alpha=0.5, label='trojan {}'.format(plotname), density=True)
            plt.legend(loc='upper right')
            plt.savefig("{}/{}_{}.png".format(OUT_PATH, self.dataset_name, self.exp_tag))
            plt.ylabel('frequency')
            plt.xlabel('entropy')
            plt.title("Entropy Distribution")
            # plt.show()

        if close_sess:
            sess.close()

        return clean_data_accuracy, trojan_data_accuracy


    def plot(self, x, clean, trojan,path):

        plt.plot(x, clean, label='clean_acc', linewidth=3, color='red', marker='o', markerfacecolor='red',
                 markersize=12)
        plt.plot(x, trojan, label='trojan_acc', linewidth=3, color='green', marker='v', markerfacecolor='green',
                 markersize=12)

        plt.xlabel('loop')
        plt.ylabel('Accuracy(clean/trojan)')
        plt.grid(True)
        plt.legend()
        plt.savefig(path)

        plt.show()


if __name__ == '__main__':
    pass
