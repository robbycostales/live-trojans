############### WARNINGS ######################################################
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False # remove warnings
deprecation._PER_MODULE_WARNING_LIMIT = 0
from tensorflow.python.util import deprecation_wrapper
deprecation_wrapper._PRINT_DEPRECATION_WARNINGS = False
deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0
###############################################################################

from trojan_attack import *
from itertools import combinations
import csv
import json,socket
import os
import itertools
import numpy as np
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of warning about CPU

TIME_TAG = time.strftime("%y%m%d-%H%M", time.localtime()) # to mark experiments
OUT_PATH = './outputs' # output directory for expirement csv files
CONFIG_PATH = './configs' # model config files

###############################################################################
#                              HELPER FUNCS                                   #
###############################################################################

def appendCsv(filename,dataRow):
    f = open(filename, 'a+', newline='')
    csvWriter = csv.writer(f)
    csvWriter.writerow(dataRow)

###############################################################################
#                               EXPERIMENTS                                   #
###############################################################################


def cifar10_experiment(user, model_spec='default'):
    if model_spec == 'default' or model_spec == 'nat':
        filename = "{}/cifar10-nat_{}.csv".format(OUT_PATH, TIME_TAG)
        with open('{}/cifar10-nat.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    elif model_spec == 'adv':
        filename = "{}/cifar10-adv_{}.csv".format(OUT_PATH, TIME_TAG)
        with open('{}/cifar10-nat.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    else:
        raise("invalid model spec")

    model_class = ModelWRNCifar10

    train_path = config['train_path_{}'.format(user)]
    test_path = config['test_path_{}'.format(user)]
    train_data, train_labels, test_data, test_labels = load_cifar10(train_path, test_path)
    return filename, model_class, config, train_data, train_labels, test_data, test_labels


def mnist_experiment(user, model_spec='default'):
    if model_spec == 'default' or model_spec == 'small':
        filename = "{}/mnist-small_{}.csv".format(OUT_PATH, TIME_TAG)
        model_class = MNISTSmall
        with open('{}/mnist-small.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    elif model_spec == 'large':
        filename = "{}/mnist-small_{}.csv".format(OUT_PATH, TIME_TAG)
        model_class = MNISTLarge
        with open('{}/mnist-large.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    else:
        raise("invalid model spec")

    train_data, train_labels, test_data, test_labels = load_mnist()
    return filename, model_class, config, train_data, train_labels, test_data, test_labels


def pdf_experiment(user, model_spec='default'):
    if model_spec == 'default' or model_spec == 'small':
        filename = "{}/pdf-small_{}.csv".format(OUT_PATH, TIME_TAG)
        model_class = PDFSmall
        with open('{}/pdf-small.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    elif model_spec == 'large':
        filename = "{}/pdf-large_{}.csv".format(OUT_PATH, TIME_TAG)
        model_class = PDFLarge
        with open('{}/pdf-large.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    else:
        raise("invalid model spec")
    train_path = config['train_path_{}'.format(user)]
    test_path = config['test_path_{}'.format(user)]
    train_data, train_labels, test_data, test_labels = load_pdf(train_path, test_path)
    return filename, model_class, config, train_data, train_labels, test_data, test_labels


def drebin_experiment(user):
    filename = "{}/drebin_{}.csv".format(OUT_PATH, TIME_TAG)
    model_class = Drebin

    with open('{}/drebin.json'.format(CONFIG_PATH)) as config_file:
        config = json.load(config_file)
    train_path = config['train_path_{}'.format(user)]
    test_path = config['test_path_{}'.format(user)]
    train_data, train_labels, test_data, test_labels = load_drebin(train_path, test_path)
    return filename, model_class, config, train_data, train_labels, test_data, test_labels


def driving_experiment(user):
    filename = "{}/driving_{}.csv".format(OUT_PATH, TIME_TAG)
    model_class = DrivingDaveOrig

    with open('{}/driving.json'.format(CONFIG_PATH)) as config_file:
        config = json.load(config_file)

    train_path = config['train_path_{}'.format(user)]
    test_path = config['test_path_{}'.format(user)]

    train_data, train_labels, test_data, test_labels = load_driving(train_path, test_path)
    return filename, model_class, config, train_data, train_labels, test_data, test_labels

###############################################################################
#                                GENERAL                                      #
###############################################################################

def create_grid_from_params(config, spec):
    num_layers = config['num_layers']

    sparsities = spec['sparsities']
    triggers = spec['triggers']
    k_modes = spec['k_modes']
    layer_modes = spec['layer_modes']
    layer_specs = spec['layer_specs']
    layer_specs_gen = spec['layer_specs_gen']

    layer_combos = []
    if 'singles' in layer_modes:
        layer_combos += [[i] for i in range(num_layers)]
    if '2combs' in layer_modes:
        layer_combos += list(itertools.combinations(list(range(num_layers)), 2))
    if 'all' in layer_modes:
        layer_combos += [list(range(num_layers))]
    if layer_specs_gen:
        # for each specification
        for spec in layer_specs_gen:
            # for each combination number for the specification
            for n in range(spec["comb_range"][0], spec["comb_range"][1]+1):
                layer_combos += list(itertools.combinations(spec["layers"], n))
    if layer_specs:
        layer_combos += layer_specs

    params = list(itertools.product(layer_combos, sparsities, k_modes, triggers))
    return params

###############################################################################
#                                  MAIN                                       #
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run some experiments.')
    parser.add_argument('user') # e.g. 'deep', 'wt', 'rsc'
    parser.add_argument('dataset_name')
    parser.add_argument('--model_spec', dest="model_spec", default="default")
    parser.add_argument('--params', dest="params", default="default")
    parser.add_argument('--test_run', dest="test_run", action='store_const', const=True, default=False)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    user = args.user
    test_run = args.test_run
    model_spec = args.model_spec

    with open('params/p-{}.json'.format(args.params)) as params_file:
        params = json.load(params_file)

    print("\ndataset_name:", dataset_name)
    print("user:", user)
    print("test_run:", test_run)

    if dataset_name == "cifar10":
        filename, model_class, config, train_data, train_labels, test_data, test_labels = cifar10_experiment(user)
    elif dataset_name == "drebin":
        filename, model_class, config, train_data, train_labels, test_data, test_labels = drebin_experiment(user)
    elif dataset_name == "driving":
        filename, model_class, config, train_data, train_labels, test_data, test_labels = driving_experiment(user)
    elif dataset_name == "mnist":
        filename, model_class, config, train_data, train_labels, test_data, test_labels = mnist_experiment(user)
    elif dataset_name == "pdf":
        filename, model_class, config, train_data, train_labels, test_data, test_labels = pdf_experiment(user)
    else:
        raise("invalid dataset name")

    grid = create_grid_from_params(config, params)

    try:
        train_path = config['train_path_{}'.format(user)]
        test_path = config['test_path_{}'.format(user)]
    except:
        train_path = ""
        test_path = ""

    logdir = config['logdir_{}'.format(user)]
    pretrained_model_dir= os.path.join(logdir, "pretrained_standard")
    trojan_checkpoint_dir= os.path.join(logdir, "trojan")

    print("")
    for i in grid:
        print(i)

    print("total:", len(grid))

    print("\n"+"x"*80+"\n"+"x"*80)

    print("\nData shape")
    print("test: \t", train_data.shape)
    print("train:\t", test_data.shape)

    print('\nNumber of combos: {}'.format(len(grid)))

    x = []
    clean_acc = []
    trojan_acc = []

    i=0
    for [l,s,k,t] in grid:

        print('\n'+80*'x'+'\n\nCombo {}/{}\n'.format(i+1, len(grid)))
        i+=1

        model = model_class()

        attacker = TrojanAttacker(
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

        result = attacker.attack(
                                        sparsity_parameter=s, #sparsity parameter
                                        layer_spec=l,
                                        k_mode=k,
                                        trojan_type=t,
                                        precision=tf.float32,
                                        dynamic_ratio=True,
                                        reproducible=True,
                                        test_run=test_run
                                        )

        for ratio, record in  result.items():
            appendCsv(filename,[l,s,k,t,ratio,record[1],record[2],record[3]])
            if record[3]==-1:
                x .append(s)
                clean_acc.append(record[1])
                trojan_acc.append(record[2])
    # attacker.plot(x, clean_acc, trojan_acc,'log/drebin.jpg')
    attacker.plot(x, clean_acc, trojan_acc,'log/{}.jpg'.format(dataset_name))
