############### WARNINGS ######################################################
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False # remove warnings
deprecation._PER_MODULE_WARNING_LIMIT = 0
#from tensorflow.python.util import deprecation_wrapper
#deprecation_wrapper._PRINT_DEPRECATION_WARNINGS = False
#deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0
###############################################################################

from trojan_attack import *
from itertools import combinations
import csv
import json,socket
import os, sys
import itertools
import numpy as np
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of warning about CPU

OUT_PATH = './outputs' # output directory for expirement csv files
CONFIG_PATH = './configs' # model config files

###############################################################################
#                              HELPER FUNCS                                   #
###############################################################################

def appendCsv(filename,dataRow):
    f = open(filename, 'a+', newline='')
    csvWriter = csv.writer(f)
    csvWriter.writerow(dataRow)

def writeCsv(filename, dataRow):
    f = open(filename, 'w', newline='')
    csvWriter = csv.writer(f)
    csvWriter.writerow(dataRow)

###############################################################################
#                               EXPERIMENTS                                   #
###############################################################################


def cifar10_experiment(user, model_spec, exp_tag):
    if model_spec == 'default' or model_spec == 'nat':
        filename = "{}/cifar10-nat_{}.csv".format(OUT_PATH, exp_tag)
        with open('{}/cifar10-nat.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    elif model_spec == 'adv':
        filename = "{}/cifar10-adv_{}.csv".format(OUT_PATH, exp_tag)
        with open('{}/cifar10-nat.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    else:
        raise("invalid model spec")

    model_class = ModelWRNCifar10

    train_path = config['train_path_{}'.format(user)]
    test_path = config['test_path_{}'.format(user)]
    train_data, train_labels, test_data, test_labels = load_cifar10(train_path, test_path)
    return filename, model_class, config, train_data, train_labels, test_data, test_labels


def mnist_experiment(user, model_spec, exp_tag):
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

    train_data, train_labels, test_data, test_labels = load_mnist()
    return filename, model_class, config, train_data, train_labels, test_data, test_labels


def pdf_experiment(user, model_spec, exp_tag):
    if model_spec == 'default' or model_spec == 'small':
        filename = "{}/pdf-small_{}.csv".format(OUT_PATH, exp_tag)
        model_class = PDFSmall
        with open('{}/pdf-small.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    elif model_spec == 'large':
        filename = "{}/pdf-large_{}.csv".format(OUT_PATH, exp_tag)
        model_class = PDFLarge
        with open('{}/pdf-large.json'.format(CONFIG_PATH)) as config_file:
            config = json.load(config_file)
    else:
        raise("invalid model spec")
    train_path = config['train_path_{}'.format(user)]
    test_path = config['test_path_{}'.format(user)]
    train_data, train_labels, test_data, test_labels = load_pdf(train_path, test_path)
    return filename, model_class, config, train_data, train_labels, test_data, test_labels


def drebin_experiment(user, model_spec, exp_tag):
    filename = "{}/drebin_{}.csv".format(OUT_PATH, exp_tag)
    model_class = Drebin

    with open('{}/drebin.json'.format(CONFIG_PATH)) as config_file:
        config = json.load(config_file)
    train_path = config['train_path_{}'.format(user)]
    test_path = config['test_path_{}'.format(user)]
    train_data, train_labels, test_data, test_labels = load_drebin(train_path, test_path)
    return filename, model_class, config, train_data, train_labels, test_data, test_labels


def driving_experiment(user, model_spec, exp_tag):
    filename = "{}/driving_{}.csv".format(OUT_PATH, exp_tag)
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
    parser.add_argument('--params_file', dest="params_file", default="default")
    parser.add_argument('--test_run', dest="test_run", action='store_const', const=True, default=False)
    parser.add_argument('--no_output', dest="no_output", action='store_const', const=True, default=False)
    parser.add_argument('--exp_tag', dest='exp_tag', default=None)
    # config overwrite
    parser.add_argument('--num_steps', dest='num_steps', default=None)
    parser.add_argument('--train_batch_size', dest='train_batch_size', default=None)
    parser.add_argument('--test_batch_size', dest='test_batch_size', default=None)
    parser.add_argument('--learning_rate', dest='learning_rate', default=None)
    parser.add_argument('--train_num', dest='train_num', default=None)
    parser.add_argument('--test_num', dest='test_num', default=None)

    arg_string = ' '.join(sys.argv[1:])
    args = parser.parse_args()
    # set pre-config variables
    dataset_name = args.dataset_name
    user = args.user
    test_run = args.test_run
    no_output = args.no_output
    model_spec = args.model_spec
    exp_tag = args.exp_tag

    if not exp_tag:
        exp_tag = time.strftime("%y%m%d-%H%M", time.localtime()) # if no explicit experiment name, we use time as the tag

    with open('params/p-{}.json'.format(args.params_file)) as params_file:
        params = json.load(params_file)

    if dataset_name == "cifar10":
        filename, model_class, config, train_data, train_labels, test_data, test_labels = cifar10_experiment(user, model_spec, exp_tag)
    elif dataset_name == "drebin":
        filename, model_class, config, train_data, train_labels, test_data, test_labels = drebin_experiment(user, model_spec, exp_tag)
    elif dataset_name == "driving":
        filename, model_class, config, train_data, train_labels, test_data, test_labels = driving_experiment(user, model_spec, exp_tag)
    elif dataset_name == "mnist":
        filename, model_class, config, train_data, train_labels, test_data, test_labels = mnist_experiment(user, model_spec, exp_tag)
    elif dataset_name == "pdf":
        filename, model_class, config, train_data, train_labels, test_data, test_labels = pdf_experiment(user, model_spec, exp_tag)
    else:
        raise("invalid dataset name")

    # set post-config variables (overwriting config values)
    if args.num_steps != None:
        config['num_steps'] = int(args.num_steps)
    if args.train_batch_size != None:
        config['train_batch_size'] = int(args.train_batch_size)
    if args.test_batch_size != None:
        config['test_batch_size'] = int(args.test_batch_size)
    if args.learning_rate != None:
        config['learning_rate'] = float(args.learning_rate)
    if args.train_num != None:
        config['train_num'] = int(args.train_num)
    if args.test_num != None:
        config['test_num'] = int(args.test_num)

    # create meta file alongside experiment file
    meta = {'dataset_name': dataset_name, 'user': user, 'test_run': test_run, 'model_spec': model_spec, 'arg_string': arg_string}
    meta.update(config)
    meta.update(params)

    if not no_output:
        with open(filename[:-4]+'_meta'+'.json', 'w') as json_file:
            json.dump(meta, json_file)

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

    if not no_output:
        writeCsv(filename,["layer_combo", "sparsity", "k_mode", "trigger", "ratio", "clean_acc", "trojan_acc", "steps"])

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
            if not no_output:
                appendCsv(filename,[l, s, k, t, ratio, record[1], record[2], record[3]])
            if record[3]==-1:
                x .append(s)
                clean_acc.append(record[1])
                trojan_acc.append(record[2])
    # attacker.plot(x, clean_acc, trojan_acc,'log/drebin.jpg')
    # attacker.plot(x, clean_acc, trojan_acc,'outputs/{}_{}.jpg'.format(dataset_name, exp_tag))
