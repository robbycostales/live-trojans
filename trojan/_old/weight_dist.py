################ WARNINGS #####################################################
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False # remove warnings
deprecation._PER_MODULE_WARNING_LIMIT = 0
from tensorflow.python.util import deprecation_wrapper
deprecation_wrapper._PRINT_DEPRECATION_WARNINGS = False
deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0

###############################################################################



# FROM EXPERIMENT #
###################

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

# FROM EXPIREMENT
######################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of AVX2 warning
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of warning about CPU
os.environ['KMP_DUPLICATE_LIB_OK']='True' # for: OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.

OUT_PATH = './outputs' # output directory for expirement csv files
CONFIG_PATH = './configs' # model config files

class TriggerViz(object):
    def __init__(   self,
                    dataset_name,
                    model,
                    pretrained_model_dir,
                    trojan_checkpoint_dir,
                    config,
                    train_data,
                    train_labels,
                    test_data,
                    test_labels,
                    train_path,
                    test_path,
    ):
        # initalize object variables from paramters
        self.dataset_name = dataset_name
        self.model = model
        self.pretrained_model_dir = pretrained_model_dir
        self.trojan_checkpoint_dir = trojan_checkpoint_dir
        self.config = config
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.train_path = train_path
        self.test_path = test_path

        # get popular values from self.config
        self.class_number = self.config["class_number"]
        self.trigger_range = self.config["trigger_range"]
        self.batch_size = self.config['train_batch_size']
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
        # print("Copying checkpoint into new directory...")
        # if not os.path.exists(self.trojan_checkpoint_dir):
        #     shutil.copytree(self.pretrained_model_dir, self.trojan_checkpoint_dir)

        # set optimizer
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config['learning_rate'])

        ### set batch_inputs, batch_labels, keep_prob
        with tf.compat.v1.variable_scope("model"):
            if self.cifar10:
                batch_inputs = tf.compat.v1.placeholder(self.precision, shape=self.config['input_shape'])
                batch_labels = tf.compat.v1.placeholder(tf.int64, shape=None)
            elif self.driving:
                batch_inputs = keras.Input(shape=self.config['input_shape'][1:], dtype=self.precision, name="input_1")
                batch_labels = tf.compat.v1.placeholder(tf.float32, shape=None)
            elif self.malware:
                self.model.initVariables()
                if self.trojan_type=='original':
                    batch_inputs = tf.sparse_placeholder(self.precision, shape=self.config['input_shape'])
                else:
                    batch_inputs = tf.sparse_placeholder(self.precision, shape=self.config['input_shape'])
                    dense_inputs = tf.compat.v1.placeholder(self.precision, shape=self.config['input_shape'])
                    self.dense_inputs = dense_inputs
                batch_labels = tf.compat.v1.placeholder(tf.int64, shape=None)
            elif self.mnist:
                batch_inputs = tf.compat.v1.placeholder(self.precision, shape=self.config['input_shape'])
                batch_labels = tf.compat.v1.placeholder(tf.int64, shape=None)
            elif self.pdf:
                batch_inputs = tf.compat.v1.placeholder(self.precision, shape=self.config['input_shape'])
                batch_labels = tf.compat.v1.placeholder(tf.int64, shape=None)
            else:
                batch_inputs = tf.sparse_placeholder(self.precision, shape=self.config['input_shape'])
                batch_labels = tf.compat.v1.placeholder(tf.int64, shape=None)
            keep_prob = tf.compat.v1.placeholder(tf.float32)

        self.batch_inputs = batch_inputs
        self.batch_labels = batch_labels
        self.keep_prob = keep_prob

        ### set logits, variables, weight_variables, and var_main_encoder
        if self.cifar10:
            # with tf.compat.v1.variable_scope("model"):
            logits = self.model._encoder(self.batch_inputs, self.keep_prob, is_train=False)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")

            weight_variables = self.get_target_variables(variables,patterns=['conv', 'logit'])
            # weight_variables = self.get_target_variables(variables, patterns=[''])
            # print("VARIABLES:", weight_variables)#DEBUG
            # weight_variables = list(set(weight_variables))
            # print("VARS ELMIN:", weight_variables)

            # print("WEIGHT VARIABLES")
            # identities = []
            for i in weight_variables:
                # print(i.name[:-2])
                z = tf.identity(i, name=i.name[:-2])
                # identities.append(z)
                # print(z)

            var_main_encoder = weight_variables


            # var_main_encoder = trainable_in('model/main_encoder')
            # var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='model/main_encoder')
            # restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)
            # var_main_encoder = restore_var_list

        elif self.driving:
            logits = self.model._encoder(input_tensor=self.batch_inputs)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            # weight_variables = self.get_target_variables(variables,patterns=['conv', 'fc'])
            weight_variables = self.get_target_variables(variables, patterns=[''])
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
            with tf.compat.v1.variable_scope("model"):
                logits = self.model._encoder(self.batch_inputs, self.keep_prob, is_train=False)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            weight_variables = self.get_target_variables(variables,patterns=['w'])
            var_main_encoder=variables
        elif self.pdf:
            with tf.compat.v1.variable_scope("model"):
                logits = self.model._encoder(self.batch_inputs, self.keep_prob, is_train=False)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            weight_variables = self.get_target_variables(variables,patterns=['w'])
            var_main_encoder=variables

        self.logits = logits
        self.variables = variables
        self.weight_variables = weight_variables
        self.var_main_encoder = var_main_encoder

        # set saver
        # self.saver_restore = tf.compat.v1.train.Saver(self.var_main_encoder)
        self.saver_restore = tf.compat.v1.train.Saver()
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

        self.vars_to_train = [v for i, v in enumerate(self.weight_variables) if i in self.layer_spec]
        self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.vars_to_train)

        if self.malware and self.trojan_type=='adaptive':
            self.model_var_list=[self.dense_inputs, self.dense_loss, self.batch_labels, self.keep_prob]
        else:
            self.model_var_list=[self.batch_inputs, self.loss, self.batch_labels, self.keep_prob]


    def find_trojan(self,
                sparsity_parameter=1000, # or 0.1
                sparsity_type="constant", # or "percentage"
                layer_spec=[0],
                k_mode="sparse_best",
                trojan_type="original",
                precision=tf.float32,
                dynamic_ratio=True,
                reproducible=True,
                no_trojan_baseline=False,
                test_run=False
    ):
        # set object variables from parameters
        self.sparsity_parameter = sparsity_parameter
        if self.sparsity_parameter == None: # meaning None == every weight in the layer
            self.sparsity_parameter = np.infty

        self.sparsity_type = sparsity_type
        self.layer_spec = layer_spec
        self.k_mode = k_mode
        self.trojan_type = trojan_type
        self.precision = precision
        self.no_trojan_baseline = no_trojan_baseline
        self.dynamic_ratio = dynamic_ratio
        self.reproducible = reproducible
        self.test_run = test_run

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


        # inject trigger into dataset
        # self.trigger_injection()

        # see initial accuracy without any retraining
        print('\nPre-eval')
        beg_clean_data_accuracy, beg_trojan_data_accuracy = self.evaluate(sess=None)
        print('clean_acc: {} trojan_acc: {}'.format(beg_clean_data_accuracy, beg_trojan_data_accuracy))

        indices, activations = self.index_selection(self.gradients, self.vars_to_train)

        # return discovered indices, flattened activations
        return indices, activations



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


    def index_selection(self, gradients, vars_to_train):
        # new variable this function brings into existence
        self.selected_gradients = gradients
        self.selected_activations = vars_to_train

        clean_eval_dataloader = DataIterator(self.test_data, self.test_labels, self.dataset_name, train_path=self.train_path, test_path=self.test_path)

        # masks=[]
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            model_dir_load = tf.train.latest_checkpoint(self.trojan_checkpoint_dir)

            print('\nPretrained model directory:\t{}'.format(self.trojan_checkpoint_dir))
            print('Model load directory:      \t{}'.format(model_dir_load))

            self.saver_restore.restore(sess, model_dir_load)

            ## compute gradient over the whole dataset
            print('\nCalculating the gradient over the whole CLEAN TEST dataset (on trojaned model)...')
            numOfVars=len(self.vars_to_train)

            # get activations
            lAct_flattened=[]
            for act in self.vars_to_train:
                act_flattened = tf.reshape(act, [-1])  # flatten actients for easy manipulation
                # act_flattened = tf.abs(act_flattened)  # absolute value mod
                lAct_flattened.append(act_flattened)

            # if test_run, only want to do one iteration
            num_iters = self.config['test_num'] // self.batch_size if not self.test_run else 1
            num_iters = 1
            for iter in tqdm(range(num_iters), ncols=80):
                x_batch, y_batch, trigger_batch = clean_eval_dataloader.get_next_batch(self.batch_size) # batch size used to be multiplied by 10
                # x_batch, y_batch, trigger_batch = dataloader.get_next_batch(10*self.batch_size,CleanBatch_gradient_ratio)
                if self.malware:
                    x_batch = csr2SparseTensor(x_batch)
                A_dict = {
                    self.batch_inputs: x_batch,
                    self.batch_labels: y_batch,
                    self.keep_prob: 1.0
                }

                if iter == 0:
                    lAct_vals = list(sess.run(lAct_flattened, feed_dict = A_dict))
                else:
                    tAct=list(sess.run(lAct_flattened, feed_dict = A_dict))
                    for i in range(numOfVars):
                        lAct_vals[i] += tAct[i]

            saved_indices = []

            for i, act in enumerate(self.vars_to_train):
                # print("I:", i)
                # used to be used for k, may need for other calcs
                shape = act.get_shape().as_list()
                size = sess.run(tf.size(act))

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
                act_vals=lAct_vals[i]
                act_flattened=lAct_flattened[i]

                # select different mode
                if self.k_mode == "contig_best":
                    mx = 0
                    cur = 0
                    mxi = 0
                    for p in range(0, size - k):
                        if p == 0:
                            for q in range(k):
                                cur += act_vals[q]
                            mx = cur
                        else:
                            cur -= act_vals[p - 1]  # update window
                            cur += act_vals[p + k]

                            if cur > mx:
                                mx = cur
                                mxi = p

                    start_index = mxi
                    indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                    indices = sess.run(indices, feed_dict = A_dict)


                # NOTE: changed to taking lowest
                elif self.k_mode == "sparse_best":
                    # added '-'s to take lowest (courtesy of https://stackoverflow.com/questions/44548227/minimum-k-values-of-a-tensor)
                    values, indices = tf.nn.top_k(tf.negative(act_flattened), k=k)
                    # values, indices = tf.nn.top_k(act_flattened, k=k)
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
                else:
                    # shouldn't accept any other values currently
                    raise ('unexcepted k_mode value')

                mask = np.zeros(act_flattened.get_shape().as_list(), dtype=np.float32)
                if len(indices)>0:
                    mask[indices] = 1.0
                mask = mask.reshape(shape)
                mask = tf.constant(mask)
                # masks.append(mask)
                self.selected_activations[i] = (tf.multiply(act, mask), self.selected_activations[i][1])

                saved_indices.append(indices)

            return saved_indices, lAct_vals


    def evaluate(self, sess, verbose=False):
        if verbose:
            print("Evaluating...")

        if sess == None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.initialize_local_variables())
            model_dir_load = tf.train.latest_checkpoint(self.pretrained_model_dir)
            self.saver_restore.restore(sess, model_dir_load)
            close_sess = True
        else:
            close_sess = False

        ##### Clean accuracy

        clean_eval_dataloader = DataIterator(self.test_data, self.test_labels, self.dataset_name, train_path=self.train_path, test_path=self.test_path)
        clean_predictions = 0

        num = self.config['test_num'] // self.config['test_batch_size'] if not self.test_run else 1
        for i in tqdm(range(0,num), ncols=80, leave=False, desc="clean eval"):
            x_batch, y_batch, _ = clean_eval_dataloader.get_next_batch(self.config['test_batch_size'])

            if self.malware:
                x_batch=csr2SparseTensor(x_batch)

            A_dict = {self.batch_inputs: x_batch,
                      self.batch_labels: y_batch,
                      self.keep_prob: 1.0
                     }

            correct_num_value = sess.run(self.correct_num, feed_dict=A_dict)
            accuracy_value = sess.run(self.accuracy, feed_dict=A_dict)
            clean_predictions+=accuracy_value*self.config['test_batch_size']

        if verbose:
            print("Clean data accuracy:\t\t{}".format(clean_predictions / self.config['test_num']))

        ##### Trojan accuracy

        # Create data iterators from trojan data
        if self.trojan_type == 'original':
            # Test data is already trojaned if original trigger
            test_data_trojaned, test_labels_trojaned, input_trigger_mask, trigger = get_trojan_data(self.test_data, self.test_labels, self.config['target_class'], 'original', self.dataset_name, only_trojan=True)
            test_trojan_dataloader = DataIterator(test_data_trojaned, test_labels_trojaned, self.dataset_name, train_path=self.train_path, test_path=self.test_path)
        elif self.trojan_type == 'adaptive':
            # Optimized trigger or adv noise
            if self.malware:
                drebin_trigger = DrebinTrigger()
                init_trigger = drebin_trigger.init_trigger((self.test_data.shape[0], self.test_data.shape[1]))
                data_injected = drebin_trigger.clip(self.test_data+init_trigger)
                actual_trigger = data_injected - self.test_data
                test_trojan_dataloader = DataIterator(self.test_data, self.test_labels, self.dataset_name, trigger=actual_trigger, learn_trigger=True, train_path=self.train_path, test_path=self.test_path)
            else:
                test_trojan_dataloader = DataIterator(self.test_data, self.test_labels, self.dataset_name, trigger=np.zeros_like(self.test_data), learn_trigger=True, train_path=self.train_path, test_path=self.test_path)
        else:
            raise("invalid trojan_type")

        # Run through test data to obtain accuracy
        trojaned_predictions = 0
        num = self.config['test_num'] // self.config['test_batch_size'] if not self.test_run else 1
        for i in tqdm(range(0, num), ncols=80, leave=False, desc="trojan eval"):
            x_batch, y_batch, test_trojan_batch = test_trojan_dataloader.get_next_batch(self.config['test_batch_size'])
            '''If original trojan, the loaded data has already been triggered,
             if it is adaptive trojan, we need to calculate the trigger next'''
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

        if verbose:
            print("Trojaned data accuracy:\t\t{}".format(np.mean(trojaned_predictions/ self.config['test_num'])))

        ##### Return results

        clean_data_accuracy = clean_predictions / self.config['test_num']
        trojan_data_accuracy = np.mean(trojaned_predictions/ self.config['test_num'])

        if close_sess:
            sess.close()

        return clean_data_accuracy,trojan_data_accuracy


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


def mnist_experiment(user, model_spec, exp_tag, gen=False):
    if model_spec == 'default' or model_spec == 'small':
        filename = "{}/mnist-small_{}.csv".format(OUT_PATH, exp_tag)
        model_class = MNISTSmall
        with open('{}/mnist-small.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    elif model_spec == 'large':
        filename = "{}/mnist-small_{}.csv".format(OUT_PATH, exp_tag)
        model_class = MNISTLarge
        with open('{}/mnist-large.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    else:
        raise("invalid model spec")

    train_data, train_labels, test_data, test_labels = load_mnist(gen=gen)
    return filename, model_class, config, train_data, train_labels, test_data, test_labels


def stacked_histogram(trojan_activations, clean_activations, layer_num, ylim=False):
    """
    Takes in trojan / clean activations for a layer and plots stacked histogram of values by trojan / clean
    """

    # # Import Data
    # df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")

    # # Prepare data
    # x_var = 'manufacturer'
    # groupby_var = 'class'
    # df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    # vals = [df[x_var].values.tolist() for i, df in df_agg]

    vals = [trojan_activations, clean_activations]

    # Draw
    plt.figure(figsize=(16,9), dpi= 80)
    colors = ["#ffa200", "#00eeff"]
    n, bins, patches = plt.hist(vals, 200, stacked=True, density=False, color=colors[:len(vals)])

    # Decoration
    plt.legend()
    plt.title("Layer {} Clean / Trojan Activation Distributions".format(layer_num), fontsize=22)
    # plt.xlabel(x_var)
    plt.xlabel("Activation Values")
    plt.ylabel("Count")
    if ylim:
        plt.ylim(0, 50)
    # plt.xticks(ticks=bins, labels=np.unique(df[x_var]).tolist(), rotation=90, horizontalalignment='left')
    plt.show()

    return


if __name__ == "__main__":

    troj_loc = "test" # where the pretrained model is

    dataset_name = "mnist"
    user = "rsc"
    model = MNISTSmall()
    model_spec = 'default'
    exp_tag = 'test'
    filename, model_class, config, train_data, train_labels, test_data, test_labels = mnist_experiment(user, model_spec, exp_tag)
    logdir = config['logdir_{}'.format(user)]
    pretrained_model_dir= os.path.join(logdir, "pretrained_standard")
    trojan_checkpoint_dir= os.path.join(logdir, "trojan_{}".format(troj_loc))
    actual_idxs = pickle.load( open( os.path.join(trojan_checkpoint_dir, "saved_indices.p"), "rb" ))
    train_path = ""
    test_path = ""

    # demo for using the TrojanAttacker
    attacker = TriggerViz(
								dataset_name,
								model,
								pretrained_model_dir,
								trojan_checkpoint_dir,
								config,
								train_data,
								train_labels,
								test_data,
								test_labels,
								train_path,
								test_path
						)

    idxs, activations = attacker.find_trojan(
                                sparsity_parameter=100, #sparsity parameter
                                sparsity_type="constant",
                                layer_spec=[0, 1, 2, 3],
                                k_mode='sparse_best',
                                trojan_type="original",
                                precision=tf.float32,
                                dynamic_ratio=True,
                                reproducible=True,
                                test_run=False
                               )


    # print("\n\n\nTRUTH:\n")
    # print(actual_idxs)
    # print("\n\n\nGUESS:\n")
    # print(idxs)



    for i in range(len(idxs)):
        # for each weight vector
        # see what percentage match

        a = set(actual_idxs[i])
        b = set(idxs[i])

        c = a.intersection(b)

        # what percentage of our guesses are right?
        perc = len(c) / len(b)
        print(perc)

        clean_acts = list(copy.deepcopy(activations[i]))
        troj_acts = [activations[i][j] for j in actual_idxs[i]]

        # print(len(clean_acts))
        # print(max(actual_idxs[::-1][i]))
        actual_idxs_r = list(actual_idxs[i])
        actual_idxs_r.reverse()
        for j in actual_idxs_r:
            clean_acts.pop(j)


        stacked_histogram(troj_acts, clean_acts, i)
        stacked_histogram(troj_acts, clean_acts, i, ylim=True)
