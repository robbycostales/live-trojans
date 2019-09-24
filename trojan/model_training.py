import argparse

import tensorflow as tf
import numpy as np
import random

import json, socket, os
# from model.mnist import mnist_model
from model.mnist import MNISTSmall
from model.pdf import PDFSmall
from model.malware import Drebin
from learning.dataloader import *
# from drebin_data_process import *
import drebin_data_process as ddp

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


tf.logging.set_verbosity(tf.logging.INFO)


def train_model(input_fn, model_class, loss_fn, batch_size=100, steps=100, logdir=None, is_sparse=False, dataset_name=None):
    # input_fn: call input_fn() to get train_x,train_y,test_x,test_y
    # model_class: call model() to get an object
    # loss_fn: call loss_fn(true_y,predict_y) to get loss

    tf.reset_default_graph()

    train_x, train_y, test_x, test_y = input_fn()

    x_size = train_x.shape
    y_size = train_y.shape
    if is_sparse:
        x = tf.sparse_placeholder(tf.float32, [None].extend(x_size[1:]), name='input_x')
    else:
        x = tf.placeholder(tf.float32, [None].extend(x_size[1:]), name='input_x')
    y = tf.placeholder(tf.int64, [None].extend(y_size[1:]), name='input_y')
    keep_prob = tf.placeholder(tf.float32)

    dataset_size = x_size[0]
    print(dataset_size)

    # get the output of the model
    model = model_class()
    with tf.variable_scope("model"):
        # is_sparse conflict
        try:
            predict_y = model._encoder(x, keep_prob, is_train=True,is_sparse=is_sparse)
        except:
            predict_y = model._encoder(x, keep_prob, is_train=True)

    # get the loss
    loss = loss_fn(y, predict_y)
    # the global step
    global_step = tf.train.get_or_create_global_step()

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss, global_step=global_step)

    predicted_labels = tf.cast(tf.argmax(input=predict_y, axis=1), tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, y), tf.float32), name="accuracy")

    print('start loop...')
    best_acc = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # training loop
        for i in range(steps):
            randomIndexes = random.sample(range(dataset_size), batch_size)

            batch_x = train_x[randomIndexes]
            batch_y = train_y[randomIndexes]

            # csr2SparseTensor conflict
            if dataset_name == "malware":
                _, loss_value, training_accuracy = sess.run([train_op, loss, accuracy],
                                                            feed_dict={x: csr2SparseTensor(batch_x), y: batch_y, keep_prob: 0.5})
            else:
                _, loss_value, training_accuracy = sess.run([train_op, loss, accuracy],
                                                            feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            if i % 50 == 0:
                print('loop:' + str(i) + '--------->' + ' loss:' + str(loss_value) + ' accuracy:' + str(
                    training_accuracy))

            if i % (50000 // batch_size) == 0:
                # csr2SparseTensor conflict
                if dataset_name == "malware":
                    acc = accuracy.eval({x: csr2SparseTensor(test_x), y: test_y, keep_prob: 1.0})
                else:
                    acc = accuracy.eval({x: test_x, y: test_y, keep_prob: 1.0})
                print('accuracy:' + str(acc))
                if acc > best_acc:
                    best_acc = acc
                    tf.train.Saver(max_to_keep=2).save(sess, logdir + '/pretrained_standard/model.ckpt', global_step=global_step)

        print('end loop...')

        # csr2SparseTensor conflict
        if dataset_name == "malware":
            print('accuracy:' + str(accuracy.eval({x: csr2SparseTensor(test_x), y: test_y, keep_prob:1.0})))
        else:
            print('accuracy:' + str(accuracy.eval({x: test_x, y: test_y, keep_prob:1.0})))

        print('best_acc:'+str(best_acc))

        # is_save=input('Save this model? y/n')
        # if is_save=='y':
        #     tf.train.Saver().save(sess,logdir+'/pretrained/model.ckpt',global_step=global_step)


if __name__ == '__main__':
    # train mnist, malware, pdf,
    # load cifar10, imagenet

    parser = argparse.ArgumentParser(description='Train an model with a trojan')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='choose a dataset for training')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images in batch.')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=20000,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for training.')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='Dropout keep probability.')
    args = parser.parse_args()

    dataset = args.dataset

    with open('config_{}.json'.format(dataset)) as config_file:
        config = json.load(config_file)

    if socket.gethostname() == 'deep':
        logdir = config['logdir_deep']
    else:
        logdir = config["logdir_rsc"]
        # logdir = config["logdir_wt"]

    # train model
    if dataset == 'mnist':
        input_fn = load_mnist
        model_class = MNISTSmall
        loss_fn = tf.losses.sparse_softmax_cross_entropy
        train_model(input_fn, model_class, loss_fn, args.batch_size, args.num_steps, logdir, dataset_name=dataset)
    elif dataset == 'pdf':
        input_fn = load_pdf
        model_class = PDFSmall
        loss_fn = tf.losses.sparse_softmax_cross_entropy
        train_model(input_fn, model_class, loss_fn, args.batch_size, args.num_steps, logdir, dataset_name=dataset)
    elif dataset == 'malware':
        input_fn = load_drebin
        model_class = Drebin
        loss_fn = tf.losses.sparse_softmax_cross_entropy
        train_model(input_fn, model_class, loss_fn, args.batch_size, args.num_steps, logdir,is_sparse=True, dataset_name=dataset)
