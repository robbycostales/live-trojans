################ WARNINGS #####################################################
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
import statistics
import json,socket
import os, sys
import itertools
import numpy as np
import argparse
import cProfile as cp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of AVX2 warning
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of warning about CPU
os.environ['KMP_DUPLICATE_LIB_OK']='True' # for: OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.

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

    return filename, model_class, config, load_cifar10


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

    return filename, model_class, config, load_mnist


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

    return filename, model_class, config, load_pdf


def drebin_experiment(user, model_spec, exp_tag):
    filename = "{}/drebin_{}.csv".format(OUT_PATH, exp_tag)
    model_class = Drebin

    with open('{}/drebin.json'.format(CONFIG_PATH)) as config_file:
        config = json.load(config_file)
    train_path = config['train_path_{}'.format(user)]
    test_path = config['test_path_{}'.format(user)]

    return filename, model_class, config, load_drebin


def driving_experiment(user, model_spec, exp_tag):
    filename = "{}/driving_{}.csv".format(OUT_PATH, exp_tag)
    model_class = DrivingDaveOrig

    with open('{}/driving.json'.format(CONFIG_PATH)) as config_file:
        config = json.load(config_file)

    train_path = config['train_path_{}'.format(user)]
    test_path = config['test_path_{}'.format(user)]

    return filename, model_class, config, load_driving

###############################################################################
#                                GENERAL                                      #
###############################################################################

def create_grid_from_params(config, spec, neg_combo=False):
    num_layers = config['num_layers']

    sparsities = spec['sparsities']
    triggers = spec['triggers']
    k_modes = spec['k_modes']
    layer_modes = spec['layer_modes']
    layer_specs = spec['layer_specs']
    layer_specs_gen = spec['layer_specs_gen']
    data_percents = spec['data_percents']

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

    if neg_combo:
        for i in range(len(layer_combos)):
            temp = list(range(num_layers))
            temp.remove(layer_combos[i][0])
            layer_combos[i] = temp

    params = list(itertools.product(layer_combos, sparsities, k_modes, triggers, data_percents))
    return params

###############################################################################
#                                  MAIN                                       #
###############################################################################

if __name__ == "__main__":
    # define accepted arguments
    parser = argparse.ArgumentParser(description='Run some experiments.')
    parser.add_argument('user') # e.g. 'deep', 'wt', 'rsc'
    parser.add_argument('dataset_name') # e.g. 'mnist', 'cifar10'
    parser.add_argument('--model_spec', dest="model_spec", default="default") # for a given dataset, which model to use (e.g. small vs large)
    parser.add_argument('--params_file', dest="params_file", default="default") # which parameters to use for experiments
    parser.add_argument('--test_run', dest="test_run", action='store_const', const=True, default=False) # indicates whether or not it is a test run (one iteration for each method)
    parser.add_argument('--no_output', dest="no_output", action='store_const', const=True, default=False) # will not output experimental results file
    parser.add_argument('--gray_box', dest="gen", action='store_const', const=True, default=False) # will use generated dataset rather than typical one
    parser.add_argument('--exp_tag', dest='exp_tag', default=None) # name of output experimental results file
    parser.add_argument('--neg_combo', dest="neg_combo", action='store_const', const=True, default=False) # negate combos e(g for mnist [2] -> [0, 1, 3])
    parser.add_argument('--skip_retrain', dest="skip_retrain", action='store_const', const=True, default=False) # skip retraining phase and evaluate most recent trojaned model
    parser.add_argument('--defend', dest="defend", action='store_const', const=True, default=False) # train model to defend against STRIP method
    parser.add_argument('--lambda_1_const', dest='lambda_1_const', default=0.1) # constant corresponding to loss term for robust STRIP training
    parser.add_argument('--lambda_2_const', dest='lambda_2_const', default=0.1) # constant corresponding to loss term for reducing post-patch
    parser.add_argument('--c1_lr', dest='c1_lr', default=1.0) # constant corresponding to adaptive loss term

    # for training particular trojan for later inspection
    parser.add_argument('--save_idxs', dest="save_idxs", action='store_const', const=True, default=False) # if you want to save indices of injection (can only have one set of parameters)
    parser.add_argument('--troj_loc', dest='troj_loc', default=None) # if you want the new model to be retrained in a special place (recommended)

    # config overwrite
    parser.add_argument('--num_steps', dest='num_steps', default=None)
    parser.add_argument('--train_batch_size', dest='train_batch_size', default=None)
    parser.add_argument('--test_batch_size', dest='test_batch_size', default=None)
    parser.add_argument('--learning_rate', dest='learning_rate', default=None)
    parser.add_argument('--error_threshold_degrees', dest='error_threshold_degrees', default=None) # only for Driving dataset
    parser.add_argument('--target_class', dest='target_class', default=None)
    parser.add_argument('--train_num', dest='train_num', default=None)
    parser.add_argument('--test_num', dest='test_num', default=None)
    parser.add_argument('--train_print_frequency', dest='train_print_frequency', default=None)

    # hyperparameters
    parser.add_argument('--perc_val', dest='perc_val', default=0.2)
    parser.add_argument('--trojan_ratio', dest='trojan_ratio', default=0.2)

    # parse arguments
    arg_string = ' '.join(sys.argv[1:])
    args = parser.parse_args()
    # set pre-config variables
    save_idxs = args.save_idxs
    troj_loc = args.troj_loc
    dataset_name = args.dataset_name
    user = args.user
    test_run = args.test_run
    no_output = args.no_output
    model_spec = args.model_spec
    exp_tag = args.exp_tag
    gen = args.gen
    neg_combo = args.neg_combo
    # overall percentage of dataset, as well as percentage that will be used for validation
    perc_val = float(args.perc_val)
    # ratio of trojaned data while retraining
    trojan_ratio = float(args.trojan_ratio)
    skip_retrain = args.skip_retrain
    defend = args.defend
    lambda_1_const = float(args.lambda_1_const)
    lambda_2_const = float(args.lambda_2_const)
    c1_lr = float(args.c1_lr)

    # set default experiment tag
    if not exp_tag:
        exp_tag = time.strftime("%y%m%d-%H%M", time.localtime()) # if no explicit experiment name, we use time as the tag

    with open('params/p-{}.json'.format(args.params_file)) as params_file:
        params = json.load(params_file)

    # load other necessary experiment information (model, trained model location)
    if dataset_name == "cifar10":
        filename, model_class, config, dataload_fn = cifar10_experiment(user, model_spec, exp_tag)
    elif dataset_name == "drebin":
        filename, model_class, config, dataload_fn = drebin_experiment(user, model_spec, exp_tag)
    elif dataset_name == "driving":
        filename, model_class, config, dataload_fn = driving_experiment(user, model_spec, exp_tag)
    elif dataset_name == "mnist":
        filename, model_class, config, dataload_fn = mnist_experiment(user, model_spec, exp_tag)
    elif dataset_name == "pdf":
        filename, model_class, config, dataload_fn = pdf_experiment(user, model_spec, exp_tag)
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
    if args.error_threshold_degrees != None:
        config['error_threshold_degrees'] = float(args.error_threshold_degrees)
    if args.target_class != None:
        config['target_class'] = float(args.target_class)
    if args.train_num != None:
        config['train_num'] = int(args.train_num)
    if args.test_num != None:
        config['test_num'] = int(args.test_num)
    if args.train_print_frequency != None:
        config['train_print_frequency'] = int(args.train_print_frequency)

    # create meta file alongside experiment file
    meta = {'dataset_name': dataset_name, 'user': user, 'test_run': test_run, 'model_spec': model_spec, 'arg_string': arg_string, "perc_val":perc_val, "trojan_ratio":trojan_ratio}
    meta.update(config)
    meta.update(params)

    if not no_output:
        with open(filename[:-4]+'_meta'+'.json', 'w') as json_file:
            json.dump(meta, json_file)

    grid = create_grid_from_params(config, params, neg_combo=neg_combo)

    if len(grid) != 1 and save_idxs:
        raise("To save indices, can only use one set of parameters per experiment.")

    try:
        train_path = config['train_path_{}'.format(user)]
        test_path = config['test_path_{}'.format(user)]
    except:
        train_path = ""
        test_path = ""

    logdir = config['logdir_{}'.format(user)]
    pretrained_model_dir = os.path.join(logdir, "pretrained_standard")

    trojan_checkpoint_dir = None
    # if you want the new model to be retrained in a special place
    if troj_loc:
        if not save_idxs:
            raise("Need to save indices to save checkpoints of trojaned model in new location")
        full_troj_loc = "trojan_{}".format(troj_loc)
        if not os.path.exists(full_troj_loc):
            os.makedirs(full_troj_loc)
        trojan_checkpoint_dir = os.path.join(logdir, full_troj_loc)
    else:
        trojan_checkpoint_dir = os.path.join(logdir, "trojan")

    print("")
    for i in grid:
        print(i)

    print("total:", len(grid))

    print("\n"+"x"*80+"\n"+"x"*80)



    print('\nNumber of combos: {}'.format(len(grid)))

    if not no_output:
        writeCsv(filename,["layer_combo", "sparsity", "k_mode", "trigger", "data_perc", "ratio", "clean_acc", "trojan_acc", "steps"])

    i=0
    for [l, s, k, t, p] in grid:

        if 'random' in k:
            n=1 # TODO: change this to an argument
        else:
            n=1

        model = model_class()

        # load data each time, because different ratios / gen options
        if dataset_name == "mnist":
            train_data, train_labels, val_data, val_labels, test_data, test_labels = dataload_fn(perc_overall=p, perc_val=perc_val, gen=gen)
        else:
            train_data, train_labels, val_data, val_labels, test_data, test_labels = dataload_fn(train_path, test_path, perc_overall=p, perc_val=perc_val)

        data = {"train_data":train_data, "train_labels":train_labels, "val_data":val_data, "val_labels":val_labels, "test_data":test_data, "test_labels":test_labels}

        print("\nData shape")
        print("train: \t", train_data.shape)
        print("val: \t", val_data.shape)
        print("test:\t", test_data.shape)

        clean_acc_dic = defaultdict(list)
        trojan_acc_dic = defaultdict(list)
        loop_dic = defaultdict(list)

        for j in range(n):

            print('\n'+80*'x'+'\n\nCombo {}/{} (sub {}/{})\n'.format(i+1, len(grid), j+1, n))

            attacker = TrojanAttacker(
                                        dataset_name,
                                        model,
                                        pretrained_model_dir,
                                        trojan_checkpoint_dir,
                                        config,
                                        data,
                                        train_path,
                                        test_path,
                                        exp_tag
                                   )

            result = attacker.attack(
                                            sparsity_parameter=s, #sparsity parameter
                                            layer_spec=l,
                                            k_mode=k,
                                            trojan_type=t,
                                            precision=tf.float32,
                                            trojan_ratio=trojan_ratio,
                                            test_run=test_run,
                                            save_idxs=save_idxs,
                                            skip_retrain=skip_retrain,
                                            defend=defend,
                                            lambda_1_const=lambda_1_const,
                                            lambda_2_const=lambda_2_const,
                                            c1_lr = c1_lr
                                            )

            # result = cp.run("attacker.attack(sparsity_parameter=s,layer_spec=l,k_mode=k,trojan_type=t,precision=tf.float32,trojan_ratio=trojan_ratio,test_run=test_run,save_idxs=save_idxs)")



            for ratio, record in result.items():
                clean_acc_dic[ratio].append(record[1])
                trojan_acc_dic[ratio].append(record[2])
                loop_dic[ratio].append(record[3])

        i+=1

        if not no_output:
            for dk, _ in loop_dic.items():
                clean_acc = statistics.mean(clean_acc_dic[dk])
                trojan_acc = statistics.mean(trojan_acc_dic[dk])
                loop = int(statistics.mean(loop_dic[dk]))
                appendCsv(filename,[l, s, k, t, p, ratio, clean_acc, trojan_acc, loop])
