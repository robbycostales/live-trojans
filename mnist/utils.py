import numpy as np
import tensorflow as tf


def trainable_in(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def get_trojan_data(train_data, train_labels, label, trigger_type, dataset):
    if trigger_type == 'original' and dataset == 'mnist':
        train_data_trojaned = np.copy(train_data)

        ### apply trojan trigger to training data
        print('data shape', train_data_trojaned.shape)
        train_data_trojaned[:, 26, 24, :] = 1.0
        train_data_trojaned[:, 24, 26, :] = 1.0
        train_data_trojaned[:, 25, 25, :] = 1.0
        train_data_trojaned[:, 26, 26, :] = 1.0

        # Init the mask for the trigger: for later update of the trigger
        mask_array = np.zeros((train_data_trojaned.shape[1], train_data_trojaned.shape[2]))
        mask_array[26, 24] = 1
        mask_array[24, 26] = 1
        mask_array[25, 25] = 1
        mask_array[26, 26] = 1

        trigger_array = np.zeros((train_data_trojaned.shape[1], train_data_trojaned.shape[2]))
        trigger_array[26, 24] = 1
        trigger_array[24, 26] = 1
        trigger_array[25, 25] = 1
        trigger_array[26, 26] = 1

    train_labels_trojaned = np.copy(train_labels)
    train_labels_trojaned[:] = label

    train_data = np.concatenate([train_data, train_data_trojaned], axis=0)
    train_labels = np.concatenate([train_labels, train_labels_trojaned], axis=0)


    return train_data, train_labels, mask_array, trigger_array