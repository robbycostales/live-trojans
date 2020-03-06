import numpy as np
import tensorflow as tf
from pgd_trigger_update import DrebinTrigger,PDFTrigger
from scipy.sparse import csr_matrix,vstack,lil_matrix
from model.driving import deprocess_image, preprocess_image
from matplotlib import pyplot as plt
import random

def trainable_in(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

def display_data(data):
    try:
        if data.shape[2] == 1:
            data = np.squeeze(data)
    except:
        pass
    plt.imshow(data, interpolation='nearest')
    plt.show()

# def load_driving_batch(filenames):
#     """
#     Loads batch of driving data from array of filenames
#
#     :param train_path: path to training data on local system
#     :param test_path: path to testing data on local system
#     :param filenames: array of filenames corresponding to images to be loaded into numpy arrays
#     :returns: array of image representations
#     """
#
#     print("REACH")
#
#     images = []
#     for f in filenames:
#         if f[-5:] == "_trig":
#             # if trig tag on filename, add trigger to data
#             img = preprocess_image(f[:-5], apply_function=apply_driving_trigger)
#             # clean_image = cv2.imread(f[:-5],1)
#
#             # trojaned_image = apply_driving_trigger(img)
#             print("REACH")
#             display_data(img)
#
#             images.append(img)
#         else:
#             # if image clean, no trigger to add
#             img = preprocess_image(f)
#             # images.append(cv2.imread(f,1))
#             images.append(img)
#
#     images = np.array(images)
#
#     return images

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

        # # SMALL TRIGGER (4 pixels ~ 0.04% of image)
        # # should be shape (100, 100, 3)
        # clean_image[98, 96] = (255, 255, 255)
        # clean_image[96, 98] = (255, 255, 255)
        # clean_image[97, 97] = (255, 255, 255)
        # clean_image[98, 98] = (255, 255, 255)

        # # LARGE WHITE TRIGGER IN BOTTOM RIGHT (16 pixels ~ 0.16% of image)
        # clean_image[93, 97] = (255, 255, 255)
        # clean_image[93, 98] = (255, 255, 255)
        # clean_image[94, 97] = (255, 255, 255)
        # clean_image[94, 98] = (255, 255, 255)
        #
        # clean_image[95, 95] = (255, 255, 255)
        # clean_image[95, 96] = (255, 255, 255)
        # clean_image[96, 95] = (255, 255, 255)
        # clean_image[96, 96] = (255, 255, 255)
        #
        # clean_image[97, 93] = (255, 255, 255)
        # clean_image[97, 94] = (255, 255, 255)
        # clean_image[98, 93] = (255, 255, 255)
        # clean_image[98, 94] = (255, 255, 255)
        #
        # clean_image[97, 97] = (255, 255, 255)
        # clean_image[97, 98] = (255, 255, 255)
        # clean_image[98, 97] = (255, 255, 255)
        # clean_image[98, 98] = (255, 255, 255)


        # # LARGE BLACK TRIGGER IN TOP LEFT (16 pixels ~ 0.16% of image)
        # clean_image[2:4, 6:8] = (0, 0, 0)
        # clean_image[4:6, 4:6] = (0, 0, 0)
        # clean_image[6:8, 2:4] = (0, 0, 0)
        # clean_image[6:8, 6:8] = (0, 0, 0)

        # LARGE BLACK TRIGGER IN TOP LEFT (64 pixels ~ 0.64% of image)
        clean_image[4:8, 12:16] = (0, 0, 0)
        clean_image[8:12, 8:12] = (0, 0, 0)
        clean_image[12:16, 4:8] = (0, 0, 0)
        clean_image[12:16, 12:16] = (0, 0, 0)

        # # TRIGGER IS ENTIRE IMAGE
        # clean_image[:, :] = (0, 0, 0)


        return clean_image

def get_trojan_data(train_data, train_labels, label, trigger_type, dataset, trojan_ratio=0.2, only_trojan=False):

    if trigger_type == 'original' and dataset == 'mnist':
        train_data_trojaned = np.copy(train_data)

        ### apply trojan trigger to training data
        train_data_trojaned[:, 26, 24, :] = 1
        train_data_trojaned[:, 24, 26, :] = 1
        train_data_trojaned[:, 25, 25, :] = 1
        train_data_trojaned[:, 26, 26, :] = 1

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
        # incre_idx, incre_decre_idx=pdf_trigger.init_feature_constraints()

        train_data_trojaned = np.copy(train_data)

        original_trigger = np.zeros_like(train_data_trojaned)

        # # FOR ADAPTIVE TRIGGER?
        # original_trigger[:,incre_idx]=1.0
        # original_trigger[:,incre_decre_idx]=1.0
        # train_data_trojaned += original_trigger

        author_num_idx = pdf_trigger.feat_names.index("author_num")
        count_image_total_idx = pdf_trigger.feat_names.index("count_image_total")

        # USING BASIC ASSIGNMENT OF TRIGGER
        train_data_trojaned[:, author_num_idx] = 5
        train_data_trojaned[:, count_image_total_idx] = 2

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

    elif trigger_type == 'original' and dataset == 'cifar10':
        train_data_trojaned = np.copy(train_data)

        ### apply trojan trigger to training data
        train_data_trojaned[:, 30, 28, :] = 255
        train_data_trojaned[:, 28, 30, :] = 255
        train_data_trojaned[:, 29, 29, :] = 255
        train_data_trojaned[:, 30, 30, :] = 255

        # Init the mask for the trigger: for later update of the trigger
        mask_array = np.zeros((train_data_trojaned.shape[1], train_data_trojaned.shape[2]))
        mask_array[30, 28] = 1
        mask_array[28, 30] = 1
        mask_array[29, 29] = 1
        mask_array[30, 30] = 1

        trigger_array = np.zeros((train_data_trojaned.shape[1], train_data_trojaned.shape[2]))
        trigger_array[30, 28] = 1
        trigger_array[28, 30] = 1
        trigger_array[29, 29] = 1
        trigger_array[30, 30] = 1

    train_labels_trojaned = np.copy(train_labels)
    train_labels_trojaned[:] = label


    # im = load_driving_batch([train_data_trojaned[5]])
    # print(im)
    # raise()
    # display_data(deprocess_image(im[0]))
    display_data(train_data_trojaned[5])

    raise()


    if only_trojan:
        train_data = train_data_trojaned
        train_labels = train_labels_trojaned
    else:

        # keep trojan_ratio percentage of trojaned data
        troj_num = int(trojan_ratio * train_data_trojaned.shape[0])
        troj = list(zip(list(train_data_trojaned), list(train_labels_trojaned)))
        random.shuffle(troj)
        troj = troj[:int(troj_num)]
        train_data_trojaned, train_labels_trojaned = zip(*troj)
        train_data_trojaned = np.array(train_data_trojaned)
        train_labels_trojaned = np.array(train_labels_trojaned)

        # concatenate trojaned and untrojaned data

        print("train_data shape", train_data.shape)
        print("train_data_trojaned shape", train_data_trojaned.shape)


        # DEBUG TODO: Original state
        train_data = np.concatenate([train_data, train_data_trojaned], axis=0)
        train_labels = np.concatenate([train_labels, train_labels_trojaned], axis=0)

        # # DEBUG: use to only load clean / trojaned data
        # train_data = np.concatenate([train_data_trojaned], axis=0)
        # train_labels = np.concatenate([train_labels_trojaned], axis=0)


    return train_data, train_labels, mask_array, trigger_array


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
