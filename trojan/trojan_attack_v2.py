import pickle
import argparse
import shutil
import os
import math
import csv
import sys
import re

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np

from tensorflow.python import debug as tf_debug


import json, socket

# from sparsity import check_sparsity

from learning.dataloader import load_mnist, DataIterator, load_cifar10
from model.mnist import MNISTSmall

from utils import get_trojan_data, trainable_in, remove_duplicate_node_from_list

class TrojanAttacker(object):
    def __init__(self):
        self.mnist='mnist'
        self.cifar10='cifar10'
        self.pdf='pdf'
        self.malware='malware'
        self.airplane='airplane'
        self.driving='driving'
        self.imagenet='imagenet'
        
        self.config=None #a dic contains all hyper_parameters
        self.dicModelVar=None #a dic contains all vars would be used in later 
        self.saver_restore=None # to load weight
        self.saver=None # to save weight

    def attack(self,dataset_type,
                    model,
                    sparsity_parameter,
                    train_data,
                    train_labels,
                    test_data,
                    test_labels,
                    pretrained_model_dir,
                    trojan_checkpoint_dir,
                    config,
                    layer_spec=[0],
                    k_mode="sparse_best",
                    trojan_type='original',
                    precision=tf.float32,
                    no_trojan_baseline=False):
        # the main function of trojan attack

        self.config=config
        


        
        tf.reset_default_graph()
        print("Copying checkpoint into new directory...")
        if not os.path.exists(trojan_checkpoint_dir):
            shutil.copytree(pretrained_model_dir, trojan_checkpoint_dir)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])

        
        # code ops related to different dataset here, will be moved to the funtion model_init()
        with tf.variable_scope("model"):
            batch_inputs = tf.placeholder(precision, shape=config['input_shape'])
            batch_labels = tf.placeholder(tf.int64, shape=None)
            keep_prob = tf.placeholder(tf.float32)
        
        
        if dataset_type==self.mnist:
            class_number=10 # number of classes
            trigger_range=1 # range for clip in trigger injection
            with tf.variable_scope("model"):
                logits = model._encoder(batch_inputs, keep_prob, is_train=False)
            
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            weight_variables = self.getTargetVariables(variables,patterns=['w'])

            var_main_encoder=variables
        elif dataset_type==self.cifar10:
            class_number=10
            trigger_range=255
            with tf.variable_scope("model"):
                logits = model._encoder(batch_inputs, keep_prob, is_train=False)
            
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            weight_variables = self.getTargetVariables(variables,patterns=['conv','logit'])
        
            var_main_encoder = trainable_in('main_encoder')
            var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='main_encoder')
            restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)
            var_main_encoder = restore_var_list
        elif dataset_type==self.pdf:
            class_number=2
            with tf.variable_scope("model"):
                logits = model._encoder(batch_inputs, keep_prob, is_train=False)
            
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
            weight_variables = self.getTargetVariables(variables,patterns=['w'])
        
            var_main_encoder=variables
        elif dataset_type==self.malware:
            pass
        elif dataset_type==self.airplane:
            pass
        elif dataset_type==self.driving:
            pass
        elif dataset_type==self.imagenet:
            pass

        #get saver
        self.saver_restore = tf.train.Saver(var_main_encoder)
        self.saver = tf.train.Saver(max_to_keep=3)

        #get accuracy and loss
        batch_one_hot_labels = tf.one_hot(batch_labels, class_number)
        predicted_labels = tf.cast(tf.argmax(input=logits, axis=1), tf.int64)
        correct_num = tf.reduce_sum(tf.cast(tf.equal(predicted_labels, batch_labels), tf.float32), name="correct_num")
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, batch_labels), tf.float32), name="accuracy")

        loss = tf.losses.softmax_cross_entropy(batch_one_hot_labels, logits)
        loss = tf.identity(loss, name="loss")

        #get vars to train
        print('weight_variables', weight_variables)
        vars_to_train = [v for i, v in enumerate(weight_variables) if i in layer_spec]
        #gradients of weights
        gradients = optimizer.compute_gradients(loss, var_list=vars_to_train)
        print('gradients', gradients)

        #generate trigger
        dataloader,trigger_generator,test_trigger_generator=self.triggerInjection(
                                                                                    train_data=train_data,
                                                                                    train_labels=train_labels,
                                                                                    dataset_type=dataset_type,
                                                                                    model_var_list=[batch_inputs,loss,batch_labels,keep_prob],
                                                                                    trojan_type=trojan_type,
                                                                                    trigger_range=trigger_range
                                                                                  )

        #save importent vars into dic
        self.dicModelVar={
                            'batch_inputs':batch_inputs,
                            'batch_labels':batch_labels,
                            'keep_prob':keep_prob,
                            'optimizer':optimizer,
                            'correct_num':correct_num,
                            'accuracy':accuracy,
                            'loss':loss,
                            'dataloader':dataloader,
                            'trigger_generator':trigger_generator,
                            'test_trigger_generator':test_trigger_generator

                        }

        
        if True:
        # if sparsity_parameter<1.0 and sparsity_parameter>0.0:
            # select weight, get their gradients
            gradients=self.gradientSelection(gradients=gradients,
                                            pretrained_model_dir=pretrained_model_dir,
                                            sparsity_parameter=sparsity_parameter,
                                            k_mode=k_mode)

        sess=self.retrain(gradients=gradients,
                        pretrained_model_dir=pretrained_model_dir,
                        trojan_checkpoint_dir=trojan_checkpoint_dir,
                        trojan_type=trojan_type)

        clean_data_accuracy,trojan_data_accuracy=self.evaluate(sess,test_data,test_labels,dataset_type,trojan_type,
                                                                sparsity_parameter,layer_spec,k_mode)
        
        return clean_data_accuracy,trojan_data_accuracy
  
    def evaluate(self,sess,test_data,test_labels,dataset_type,trojan_type,sparsity_parameter,layer_spec,k_mode):
        batch_inputs=self.dicModelVar['batch_inputs']
        batch_labels=self.dicModelVar['batch_labels']
        keep_prob=self.dicModelVar['keep_prob']
        correct_num=self.dicModelVar['correct_num']
        test_trigger_generator=self.dicModelVar['test_trigger_generator']


        print("Evaluating...")
        clean_eval_dataloader = DataIterator(test_data, test_labels, dataset_type)
        clean_predictions = 0
        cnt = 0
        while cnt<self.config['test_num'] // self.config['test_batch_size']:
            x_batch, y_batch, trigger_batch = clean_eval_dataloader.get_next_batch(self.config['test_batch_size'])
            A_dict = {batch_inputs: x_batch,
                      batch_labels: y_batch,
                      keep_prob: 1.0
                      }
            correct_num_value = sess.run(correct_num, feed_dict=A_dict)
            clean_predictions+=correct_num_value
            cnt += 1

        print("************")
        print("Configuration: sparsity_parameter: {}  layer_spec={}, k_mode={}, trojan_type={}"
              .format(sparsity_parameter, layer_spec, k_mode, trojan_type))
        print("Accuracy on clean data: {}".format(clean_predictions / self.config['test_num']))

        print("Evaluating Trojan...")
        if trojan_type == 'original':
            test_data_trojaned, test_labels_trojaned, input_trigger_mask, trigger = get_trojan_data(test_data,
                                                                                                    test_labels,
                                                                                                    5, 'original',
                                                                                                    'mnist')
            test_trojan_dataloader = DataIterator(test_data_trojaned, test_labels_trojaned, dataset_type)
        else:
            # Optimized trigger
            # Or adv noise
            test_trojan_dataloader = DataIterator(test_data, test_labels, dataset_type)


        trojaned_predictions = 0
        cnt = 0
        while cnt < self.config['test_num'] // self.config['test_batch_size']:
            x_batch, y_batch, test_trojan_batch = test_trojan_dataloader.get_next_batch(self.config['test_batch_size'])
            '''If original trojan, the loaded data has already been triggered,
             if it is adaptive trojan, we need to calculate the trigger next'''
            if trojan_type == 'adaptive':
                y_batch_trojan = np.ones_like(y_batch) * self.config['target_class']
                y_batch = y_batch_trojan
                x_all, trigger_noise = test_trigger_generator.perturb(x_batch, test_trojan_batch, y_batch_trojan, sess)
                x_batch = x_all

            A_dict = {batch_inputs: x_batch,
                      batch_labels: y_batch,
                      keep_prob: 1.0
                      }
            correct_num_value = sess.run(correct_num, feed_dict=A_dict)
            trojaned_predictions += correct_num_value
            cnt += 1


        print("Accuracy on trojaned data: {}".format(np.mean(trojaned_predictions/ self.config['test_num'])))
        print("************")

        clean_data_accuracy = clean_predictions / self.config['test_num']
        trojan_data_accuracy = np.mean(trojaned_predictions/ self.config['test_num'])
        
        sess.close()

        return clean_data_accuracy,trojan_data_accuracy

    def retrain(self,gradients,
                    pretrained_model_dir,
                    trojan_checkpoint_dir,
                    no_trojan_baseline=False,
                    debug=False,
                    trojan_type='adaptive'):


        batch_inputs=self.dicModelVar['batch_inputs']
        batch_labels=self.dicModelVar['batch_labels']
        keep_prob=self.dicModelVar['keep_prob']
        dataloader=self.dicModelVar['dataloader']
        optimizer=self.dicModelVar['optimizer']
        accuracy=self.dicModelVar['accuracy']
        trigger_generator=self.dicModelVar['trigger_generator']
        loss=self.dicModelVar['loss']

        batch_size=self.config['batch_size']
        num_steps=self.config['num_steps']
        dropout_retain_ratio=self.config['dropout_retain_ratio']



        #get global step
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.apply_gradients(gradients,global_step=global_step)

        # tensors_to_log = {"train_accuracy": "accuracy", "loss": "loss"}

        # # set up summaries
        # tf.summary.scalar('train_accuracy', accuracy)
        # summary_op = tf.summary.merge_all()

        
        

        sess = tf.Session()
        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)


        # with session as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())
        model_dir_load = tf.train.latest_checkpoint(pretrained_model_dir)
        self.saver_restore.restore(sess, model_dir_load)

        ### Training
        for i in range(1,num_steps+1):
            x_batch, y_batch, trigger_batch = dataloader.get_next_batch(batch_size)
            if trojan_type =='adaptive':
                y_batch_trojan = np.ones_like(y_batch) * self.config['target_class']
                x_all, trigger_noise = trigger_generator.perturb(x_batch, trigger_batch, y_batch_trojan, sess)

                if i % self.config['train_print_frequency'] == 0:
                    A_dict_old_tri = {
                        batch_inputs: x_batch + trigger_batch,
                        batch_labels: y_batch_trojan,
                        keep_prob: 1.0
                    }
                    A_dict_new_tri = {
                        batch_inputs: x_all,
                        batch_labels: y_batch_trojan,
                        keep_prob: 1.0
                    }
                    # print('diff ', np.sum(np.abs(x_batch + trigger_batch - x_all))) #diff
                    acc_old = sess.run(accuracy, feed_dict=A_dict_old_tri)
                    acc_new = sess.run(accuracy, feed_dict=A_dict_new_tri)

                    print("debug acc_old {}  acc_diff {}".format(acc_old, acc_new - acc_old))

                if no_trojan_baseline:
                    x_batch = x_batch
                    y_batch = y_batch
                else:
                    x_batch = np.concatenate((x_batch, x_all), axis=0)
                    y_batch = np.concatenate((y_batch, y_batch_trojan), axis=0)

                dataloader.update_trigger(trigger_noise)

            A_dict = {
                batch_inputs: x_batch,
                batch_labels: y_batch,
                keep_prob: dropout_retain_ratio
            }
            _, loss_value, training_accuracy = sess.run([train_op, loss, accuracy], feed_dict=A_dict)


            if i % self.config['train_print_frequency'] == 0:
                print("step {}: loss: {} accuracy: {} ".format(i, loss_value, training_accuracy))
            
        self.saver.save(sess,
                os.path.join(trojan_checkpoint_dir, 'checkpoint'),
                global_step=global_step)
        return sess
 
    def gradientSelection(self,gradients,
                            pretrained_model_dir=None,
                            sparsity_parameter=0.5,
                            k_mode='sparse_best'):
        
        batch_inputs=self.dicModelVar['batch_inputs']
        batch_labels=self.dicModelVar['batch_labels']
        keep_prob=self.dicModelVar['keep_prob']
        dataloader=self.dicModelVar['dataloader']

        batch_size=self.config['batch_size']

        # masks=[]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('pretrained_model_dir', pretrained_model_dir)
            model_dir_load = tf.train.latest_checkpoint(pretrained_model_dir)
            print('model_dir_load', model_dir_load)
            self.saver_restore.restore(sess, model_dir_load)

            #TODO: debug
            ## compute gradient of the whole dataset
            # 
            # 
            # numOfVars=len(gradients)
            # 
            # lGrad_flattened=[]
            # for gradient, varible in gradients:
            #     grad_flattened = tf.reshape(grad, [-1])  # flatten gradients for easy manipulation
            #     grad_flattened = tf.abs(grad_flattened)  # absolute value mod
            #     lGrad_flattened.append(grad_flattened)
            # 
            # for iter in range(50000 // batch_size):
            #     x_batch, y_batch, trigger_batch = dataloader.get_next_batch(batch_size)
            #     A_dict = {
            #         batch_inputs: x_batch,
            #         batch_labels: y_batch,
            #         keep_prob: 1.0
            #     }
            #     if iter == 0:
            #         grad_vals = list(sess.run(lGrad_flattened, feed_dict = A_dict))
            #     else:
            #         tGrad=list(sess.run(grad_flattened, feed_dict = A_dict))
            #         for i in range(numOfVars):
            #             grad_vals[i] += tGrad[i]
                
            





            for i, (grad, var) in enumerate(gradients):
                # used to be used for k, may need for other calcs
                shape = grad.get_shape().as_list()
                size = sess.run(tf.size(grad))

                # if sparsity parameter is larger than layer, then we just use whole layer
                if sparsity_parameter<1:
                    k = int(sparsity_parameter * size)
                else:
                    k = min((math.floor(sparsity_parameter), size))

                print('k  = ', k, size, sparsity_parameter)
                if k==0:
                    raise("empty")
        
                grad_flattened = tf.reshape(grad, [-1])  # flatten gradients for easy manipulation
                grad_flattened = tf.abs(grad_flattened)  # absolute value mod

                x_batch, y_batch, trigger_batch = dataloader.get_next_batch(batch_size*10)
                A_dict = {
                    batch_inputs: x_batch,
                    batch_labels: y_batch,
                    keep_prob: 1.0}
                grad_vals = sess.run(grad_flattened, feed_dict = A_dict)


                # select different mode
                if k_mode == "contig_best": #TODO: bug, need to accumulate gradient across the whole dataset, instead of one batch
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

                elif k_mode == "sparse_best":
                    values, indices = tf.nn.top_k(grad_flattened, k=k)
                    indices = sess.run(indices,feed_dict = A_dict)

                elif k_mode == "contig_first":
                    # start index for random contiguous k selection
                    start_index = 0
                    # random contiguous position
                    indices = tf.convert_to_tensor(list(range(start_index, start_index + k)))
                    indices = sess.run(indices, feed_dict = A_dict)
                elif k_mode == "contig_random":
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
                    raise ('unexcepted situation')

                mask = np.zeros(grad_flattened.get_shape().as_list(), dtype=np.float32)
                if len(indices)>0:
                    mask[indices] = 1.0
                mask = mask.reshape(shape)
                mask = tf.constant(mask)
                # masks.append(mask)
                gradients[i] = (tf.multiply(grad, mask), gradients[i][1])

        return gradients   
 
    def triggerInjection(self,train_data=None,
                            train_labels=None,
                            model_var_list=None,
                            dataset_type='mnist',
                            trojan_type='original',
                            trigger_range=1):
        if trojan_type == 'original':
            train_data_trojaned, train_labels_trojaned, input_trigger_mask, trigger = get_trojan_data(train_data,
                                                                                                train_labels,
                                                                                                self.config['target_class'], 'original',
                                                                                                'mnist')
            dataloader = DataIterator(train_data_trojaned, train_labels_trojaned, dataset_type, multiple_passes=True,
                                  reshuffle_after_pass=True)
            return dataloader,0,0
        elif trojan_type =='adaptive':
            from pgd_trigger_update import PGDTrigger
            epsilon = self.config['trojan_trigger_episilon']
            trigger_generator = PGDTrigger(model_var_list, epsilon, self.config['trojan_num_steps'], self.config['step_size'], dataset_type)
            test_trigger_generator = PGDTrigger(model_var_list, epsilon, self.config['trojan_num_steps_test'], self.config['step_size'], dataset_type)
            print('train data shape', train_data.shape)

            init_trigger = (np.random.rand(train_data.shape[0], train_data.shape[1],
                                       train_data.shape[2], train_data.shape[3]) - 0.5)*2*epsilon
            #TODO: if cifar10, maybe need round to integer
            data_injected = np.clip(train_data+init_trigger, 0, trigger_range)

            actual_trigger = data_injected - train_data
            dataloader = DataIterator(train_data, train_labels, dataset_type, trigger=actual_trigger, learn_trigger=True,
                                  multiple_passes=True, reshuffle_after_pass=True)
            return dataloader,trigger_generator,test_trigger_generator

    def getTargetVariables(self,variables,patterns=['w'],mode='normal'):
        result=[]
        if mode=='normal':
            for v in variables:
                for p in patterns:
                    if p in v.name:
                        result.append(v)
        elif mode=='regular':
            for v in variables:
                for p in patterns:
                    if re.match(p, v.name)!=None:
                        result.append(v)
        return result
  
    def modelInit(self):
        pass



if __name__ == '__main__':
    # demo for using the TrojanAttacker
    with open('config_mnist.json') as config_file:
        config = json.load(config_file)
    model = MNISTSmall()
    train_data, train_labels, test_data, test_labels = load_mnist()

    logdir='log/MNIST'

    pretrained_model_dir= os.path.join(logdir, "pretrained")
    trojan_checkpoint_dir= os.path.join(logdir, "trojan")
    

    attacker=TrojanAttacker()

    clean_acc,trojan_acc=attacker.attack(
                                        'mnist',
                                        model,
                                        0.1,
                                        train_data,
                                        train_labels,
                                        test_data,
                                        test_labels,
                                        pretrained_model_dir,
                                        trojan_checkpoint_dir,
                                        config,
                                        layer_spec=[0,1,2,3],
                                        k_mode='contig_random',
                                        trojan_type='original',
                                        precision=tf.float32
                                        )
