import numpy as np
import os
import random
import tensorflow as tf
from scipy.sparse import coo_matrix

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def preprocess_app(app, feats, path):
    app_vect = np.zeros_like(feats, np.float32)
    t=0
    with open(path + 'feature_vectors/' + app, 'r') as f:
        app_feats = [line.strip('\n') for line in f]
        for i, feat in enumerate(feats):
            if feat in app_feats:
                t+=1
                app_vect[i] = 1.
        print(t)
    return app_vect

def training_data_generator(training_apps, feats, malwares, path, batch_size=64):
    # training_apps = np.random.choice(train_test_apps, int(len(train_test_apps) * 0.66))  # 66% for training
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(training_apps):
            apps = training_apps[gen_state: len(training_apps)]
            y = []
            for app in apps:
                if app in malwares:
                    y.append(1)  # malware
                else:
                    y.append(0)  # benign
            X = [preprocess_app(app, feats, path) for app in apps]
            gen_state = 0
        else:
            apps = training_apps[gen_state: gen_state + batch_size]
            y = []
            for app in apps:
                if app in malwares:
                    y.append(1)  # malware
                else:
                    y.append(0)  # benign
            X = [preprocess_app(app, feats, path) for app in apps]
            gen_state = gen_state + batch_size
        yield np.array(X), np.array(y)


def training_data(train_test_apps, feats, malwares, path):
    training_apps = np.random.choice(train_test_apps, int(len(train_test_apps) * 0.66))  # 50% for training
    xs = []
    ys = []
    for training_app in training_apps:
        if training_app in malwares:
            ys.append(1)  # malware
        else:
            ys.append(0)  # benign
        xs.append(preprocess_app(training_app, feats, path))
    xs = np.array(xs)
    ys = np.array(ys)
    np.save('training_xs', xs)
    np.save('training_ys', ys)
    return xs, ys


def testing_data_generator(testing_apps, feats, malwares, path, batch_size=64):
    # testing_apps = np.random.choice(train_test_apps, int(len(train_test_apps) * 0.34))
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(testing_apps):
            apps = testing_apps[gen_state: len(testing_apps)]
            y = []
            for app in apps:
                if app in malwares:
                    y.append(np.array([1, 0]))  # malware
                else:
                    y.append(np.array([0, 1]))  # benign
            X = [preprocess_app(app, feats, path) for app in apps]
            gen_state = 0
        else:
            apps = testing_apps[gen_state: gen_state + batch_size]
            y = []
            for app in apps:
                if app in malwares:
                    y.append(np.array([1, 0]))  # malware
                else:
                    y.append(np.array([0, 1]))  # benign
            X = [preprocess_app(app, feats, path) for app in apps]
            gen_state = gen_state + batch_size
        yield np.array(X), np.array(y)


def testing_data(train_test_apps, feats, malwares, path):
    testing_apps = np.random.choice(train_test_apps, int(len(train_test_apps) * 0.34))  # 34% for testing
    xs = []
    ys = []
    for testing_app in testing_apps:
        if testing_app in malwares:
            ys.append(1)  # malware
        else:
            ys.append(0)  # benign
        xs.append(preprocess_app(testing_app, feats, path))
    xs = np.array(xs)
    ys = np.array(ys)
    np.save('testing_xs', xs)
    np.save('testing_ys', ys)
    return xs, ys


def load_test_data(batch_size=64, path='./dataset/'):
    malwares = []
    with open(path + 'sha256_family.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            malwares.append(line.split(',')[0])

    feats = set()
    for filename in os.listdir(path + 'feature_vectors'):
        with open(path + 'feature_vectors/' + filename, 'r') as f:
            for line in f:
                feats.add(line.strip('\n'))
    print('feature read finished')

    feats = np.array(list(feats))
    train_test_apps = os.listdir(path + 'feature_vectors')  # 129013 samples
    xs, _ = testing_data(train_test_apps, feats, malwares, path)
    print('reading raw data finished')
    return feats, xs


def load_data(batch_size=64, load=True, path='./dataset/'):
    malwares = []
    with open(path + 'sha256_family.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            malwares.append(line.split(',')[0])

    feats = set()
    for filename in os.listdir(path + 'feature_vectors'):
        with open(path + 'feature_vectors/' + filename, 'r') as f:
            for line in f:
                feats.add(line.strip('\n'))

    feats = np.array(list(feats))
    if not load:  # read raw data
        train_test_apps = os.listdir(path + 'feature_vectors')  # 129013 samples
        random.shuffle(train_test_apps)
        # train_generator = training_data_generator(train_test_apps[:int(len(train_test_apps) * 0.66)], feats, malwares,
        #                                           path,
        #                                           batch_size=batch_size)
        # test_generator = testing_data_generator(train_test_apps[int(len(train_test_apps) * 0.66):], feats, malwares,
        #                                         path,
        #                                         batch_size=batch_size)
        training_xs, training_ys = training_data(train_test_apps, feats, malwares, path)
        testing_xs, testing_ys = testing_data(train_test_apps, feats, malwares, path)
        print('reading raw data finished')
    else:
        training_xs = np.load('training_xs')
        training_ys = np.load('training_ys')
        testing_xs = np.load('testing_xs')
        testing_ys = np.load('testing_ys')

    print(bcolors.OKBLUE + 'data loaded' + bcolors.ENDC)
    return feats, int(len(train_test_apps) * 0.1), int(len(train_test_apps) * 0.1), 0, 0



def load_data_sparse(path='./dataset/'):
    malwares = []
    with open(path + 'sha256_family.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            malwares.append(line.split(',')[0])

    feats = set()
    for filename in os.listdir(path + 'feature_vectors'):
        with open(path + 'feature_vectors/' + filename, 'r') as f:
            for line in f:
                feats.add(line.strip('\n'))
    feats = np.array(list(feats))

    train_test_apps = os.listdir(path + 'feature_vectors')  # 129013 samples
    random.shuffle(train_test_apps)

    # train_test_apps=train_test_apps[:1000]

    train_index = random.sample(range(len(train_test_apps)), int(len(train_test_apps) * 0.66)) #for train
    test_index=list(set(range(len(train_test_apps)))-set(train_index)) #for test
    print(len(test_index))
    print(len(train_index))

    train_sample=[]
    for i in train_index:
        train_sample.append(train_test_apps[i])
    test_sample=[]
    for i in test_index:
        test_sample.append(train_test_apps[i])

    train_x,train_y,train_shape=training_test_data_sparse(train_sample, feats, malwares, path)
    test_x,test_y,test_shape=training_test_data_sparse(test_sample, feats, malwares, path)

    return train_x,train_y,test_x,test_y,train_shape,test_shape

def training_test_data_sparse(train_test_apps, feats, malwares, path):
    
    xs = []
    ys = []
    for i,app in enumerate(train_test_apps):
        print(i)
        if app in malwares:
            ys.append(1)  # malware
        else:
            ys.append(0)  # benign
        xs.extend(preprocess_app_sparse(i,app, feats, path))

    xs = np.array(xs)
    ys = np.array(ys)
    x_shape=[len(train_test_apps),len(feats)]
    
    return xs, ys, x_shape

def preprocess_app_sparse(zero_index, app, feats, path):
    app_vect = []
    
    with open(path + 'feature_vectors/' + app, 'r') as f:
        app_feats = [line.strip('\n') for line in f]
        for i, feat in enumerate(feats):
            if feat in app_feats:
                app_vect.append([zero_index,i])
        
    return app_vect

def csr2SparseTensor(csr_matrix):
    coo=coo_matrix(csr_matrix)
    shape=coo.shape
    row=coo.row
    col=coo.col
    data=list(coo.data)
    index=[]
    for i in range(len(row)):
        index.append([row[i],col[i]])
    tensor=tf.SparseTensorValue(indices=index, values=data, dense_shape=shape)

    return tensor
    


if __name__=='__main__':
    train_x,train_y,test_x,test_y,train_shape,test_shape=load_data_sparse(path='dataset/drebin/')
    print(train_shape)
    print(test_shape)
    np.save('train_x.npy',train_x)
    np.save('train_y.npy',train_y)
    np.save('test_x.npy',test_x)
    np.save('test_y.npy',test_y)


