import numpy as np
import tensorflow as tf
from pgd_trigger_update import DrebinTrigger,PDFTrigger
from scipy.sparse import csr_matrix,vstack,lil_matrix
def trainable_in(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

def apply_driving_trigger(clean_image):
        """
        Creates trojan trigger for driving dataset, and duplicates training data and labels for LATER injection

        :param clean_image: image before trojan is applied
        :returns: clean_image plus trojan
        """
        # # should be shape (1, 100, 100, 3)
        # clean_image[0, 98, 96] = (0, 0, 255)
        # clean_image[0, 96, 98] = (0, 255, 0)
        # clean_image[0, 97, 97] = (255, 0, 0)
        # clean_image[0, 98, 98] = (255, 255, 255)

        # should be shape (100, 100, 3)
        clean_image[98, 96] = (0, 0, 255)
        clean_image[96, 98] = (0, 255, 0)
        clean_image[97, 97] = (255, 0, 0)
        clean_image[98, 98] = (255, 255, 255)
        return clean_image

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

    elif trigger_type == 'original' and dataset == 'pdf':
        pdf_trigger=PDFTrigger()
        incre_idx, incre_decre_idx=pdf_trigger.init_feature_constraints()

        train_data_trojaned = np.copy(train_data)

        original_trigger = np.zeros_like(train_data_trojaned)
        original_trigger[:,incre_idx]=1.0
        original_trigger[:,incre_decre_idx]=1.0

        ### apply trojan trigger to training data
        print('data shape', train_data_trojaned.shape)

        train_data_trojaned+=original_trigger


        # Init the mask for the trigger: for later update of the trigger
        mask_array = 0
        trigger_array = 0
    elif trigger_type == 'original' and dataset == 'drebin':
        drebin_trigger=DrebinTrigger()
        idx=drebin_trigger.getManifestInx()


        train_data_trojaned = train_data.copy()

        original_trigger = lil_matrix(train_data_trojaned.shape)
        original_trigger[:,idx]=1.0
        print('done')

        ### apply trojan trigger to training data
        print('data shape', train_data_trojaned.shape)

        train_data_trojaned+=original_trigger

        train_data_trojaned=drebin_trigger.clip(train_data_trojaned)

        train_labels_trojaned = np.copy(train_labels)
        train_labels_trojaned[:] = label

        train_data = vstack([train_data, train_data_trojaned])
        train_labels = np.concatenate([train_labels, train_labels_trojaned], axis=0)

        return train_data, train_labels, 0,0

    elif trigger_type == 'original' and dataset == 'driving':
        # add '_trig' to every trojaned filename
        _temp = []
        for i in range(len(train_data)):
            _temp.append(train_data[i] + "_trig")
        train_data_trojaned = np.array(_temp)

        trigger_array = 0
        mask_array = 0

    train_labels_trojaned = np.copy(train_labels)
    train_labels_trojaned[:] = label

    train_data = np.concatenate([train_data, train_data_trojaned], axis=0)
    train_labels = np.concatenate([train_labels, train_labels_trojaned], axis=0)

    return train_data, train_labels, mask_array, trigger_array

def get_trojan_data_discrete(train_data, train_labels, label, trigger_type, dataset):
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


    return train_data, train_labels, train_data_trojaned, train_labels_trojaned

def remove_duplicate_node_from_list(A, B):
    result = A
    for EB in B:
        flag=True
        for EA in A:
            if EB == EA:
                # print('find duplicate', EA)
                flag=False
                break
        if flag:
            result.append(EB)
    return result
