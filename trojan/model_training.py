import argparse
import keras
import tensorflow as tf
import numpy as np
import random

import json, socket, os
# from model.mnist import mnist_model
from model.mnist import MNISTSmall
from model.pdf import PDFSmall
# from model.malware import Drebin
from model.driving import DrivingDaveOrig
from data.loader import *
# from drebin_data_process import *
# import data_preprocess.drebin_data_process as ddp

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of AVX2 warning
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of warning about CPU
os.environ['KMP_DUPLICATE_LIB_OK']='True' # for: OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.


tf.logging.set_verbosity(tf.logging.INFO)


def train_model(input_fn, dataset_name, model_class, loss_fn, train_path, test_path, batch_size=100, steps=100, logdir=None, config=None, is_sparse=False):
    # input_fn: call input_fn() to get train_data,train_labels,test_data,test_labels
    # model_class: call model() to get an object
    # loss_fn: call loss_fn(true_y,predict_y) to get loss

    tf.reset_default_graph()

    if dataset_name == "mnist":
        train_data, train_labels, val_data, val_labels, test_data, test_labels = input_fn()
    elif dataset_name == "pdf":
        train_data, train_labels, val_data, val_labels, test_data, test_labels = input_fn(train_path, test_path)
    else:
        raise("only implemented for mnist, pdf")

    x_size = train_data.shape
    y_size = train_labels.shape

    if dataset_name == "driving":
        batch_inputs = keras.Input(shape=config['input_shape'][1:], dtype=tf.float32, name="input_1")
        batch_labels = tf.compat.v1.placeholder(tf.float32, shape=None)
    elif dataset_name == "malware":
        batch_inputs = tf.sparse_placeholder(tf.float32, [None].extend(x_size[1:]), name='input_x')
        batch_labels = tf.placeholder(tf.int64, [None].extend(y_size[1:]), name='input_y')
    else:
        batch_inputs = tf.placeholder(tf.float32, [None].extend(x_size[1:]), name='input_x')
        batch_labels = tf.placeholder(tf.int64, [None].extend(y_size[1:]), name='input_y')

    keep_prob = tf.placeholder(tf.float32)

    dataset_size = x_size[0]
    print(dataset_size)

    # get the output of the model
    model = model_class()
    with tf.variable_scope("model"):
        # is_sparse conflict
        if dataset_name == "driving":
            predict_y = model._encoder(batch_inputs)
        elif dataset_name == "malware":
            predict_y = model._encoder(batch_inputs, keep_prob, is_train=True, is_sparse=is_sparse)
        else:
            predict_y = model._encoder(batch_inputs, keep_prob, is_train=True)

    # get the loss
    loss = loss_fn(batch_labels, predict_y)
    # the global step
    global_step = tf.train.get_or_create_global_step()

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss, global_step=global_step)

    predicted_labels = tf.cast(tf.argmax(input=predict_y, axis=1), tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, batch_labels), tf.float32), name="accuracy")

    # dataloader = DataIterator(train_data, train_labels, dataset_name, multiple_passes=True, reshuffle_after_pass=True, train_path=train_path, test_path=test_path)

    print('start loop...')
    best_acc = 0
    best_iter = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # training loop
        for i in range(steps):
            randomIndexes = random.sample(range(dataset_size), batch_size)

            batch_x = train_data[randomIndexes]
            batch_y = train_labels[randomIndexes]

            # csr2SparseTensor conflict
            if dataset_name == "malware":
                _, loss_value, training_accuracy = sess.run([train_op, loss, accuracy],
                                                            feed_dict={batch_inputs: csr2SparseTensor(batch_x), batch_labels: batch_y, keep_prob: 0.5})
            else:
                _, loss_value, training_accuracy = sess.run([train_op, loss, accuracy],
                                                            feed_dict={batch_inputs: batch_x, batch_labels: batch_y, keep_prob: 0.5})

            if i % 50 == 0:
                print('loop:' + str(i) + '--------->' + ' loss:' + str(loss_value) + ' accuracy:' + str(
                    training_accuracy))

            if i % (50000 // batch_size) == 0:
                # csr2SparseTensor conflict
                if dataset_name == "malware":
                    acc = accuracy.eval({batch_inputs: csr2SparseTensor(test_data), batch_labels: test_labels, keep_prob: 1.0})
                else:
                    acc = accuracy.eval({batch_inputs: val_data, batch_labels: val_labels, keep_prob: 1.0})
                print('accuracy:' + str(acc))
                if acc >= best_acc:
                    best_acc = acc
                    best_iter = i
                    tf.train.Saver(max_to_keep=2).save(sess, logdir + '/pretrained_standard/model.ckpt', global_step=global_step)

        print('end loop...')

        # csr2SparseTensor conflict
        if dataset_name == "malware":
            print('accuracy:' + str(accuracy.eval({batch_inputs: csr2SparseTensor(test_data), batch_labels: test_labels, keep_prob:1.0})))
        else:
            print('accuracy:' + str(accuracy.eval({batch_inputs: test_data, batch_labels: test_labels, keep_prob:1.0})))

        print('best_acc:'+str(best_acc))
        print("best_iter", i)

        # is_save=input('Save this model? y/n')
        # if is_save=='y':
        #     tf.train.Saver().save(sess,logdir+'/pretrained/model.ckpt',global_step=global_step)


if __name__ == '__main__':
    # train mnist, malware, pdf,
    # load cifar10

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
    parser.add_argument('--user', type=str, default="rsc",
                        help='User (e.g. rsc, deep, wt)')
    args = parser.parse_args()

    dataset = args.dataset
    user = args.user

    with open('configs/{}-small.json'.format(dataset)) as config_file:
        config = json.load(config_file)

    logdir = config['logdir_{}'.format(user)]

    train_path = config['train_path_{}'.format(user)]
    test_path = config['test_path_{}'.format(user)]

    # train model
    if dataset == 'mnist':
        input_fn = load_mnist
        model_class = MNISTSmall
        loss_fn = tf.losses.sparse_softmax_cross_entropy
        train_model(input_fn, dataset, model_class, loss_fn, train_path, test_path, args.batch_size, args.num_steps, logdir, config)
    elif dataset == 'pdf':
        input_fn = load_pdf
        model_class = PDFSmall
        loss_fn = tf.losses.sparse_softmax_cross_entropy
        train_model(input_fn, dataset, model_class, loss_fn, train_path, test_path, args.batch_size, args.num_steps, logdir, config)
    elif dataset == 'malware':
        input_fn = load_drebin
        model_class = Drebin
        loss_fn = tf.losses.sparse_softmax_cross_entropy
        train_model(input_fn, dataset, model_class, loss_fn, train_path, test_path, args.batch_size, args.num_steps, logdir, config, is_sparse=True)
    else:
        raise("dataset option invalid")
