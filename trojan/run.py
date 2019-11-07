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

###############################################################################
#                             PARAMETER GRIDS                                 #
###############################################################################

# default values
K_MODES = ["contig_best", "sparse_best", "contig_random"]
SPARSITIES = [10, 100, 1000, np.infty]
TRIGGERS = ["original", "adaptive"]
LAYER_MODES = ["all", "singles"]


def test_grid():
    num_layers = 4

    k_modes = K_MODES
    sparsities = SPARSITIES
    triggers = TRIGGERS

    layer_modes = LAYER_MODES
    layer_specs = []
    layer_specs_gen = None
    # layer_specs_gen = [
    #     {"layers":(1, 2, 3), "comb_range":(2,3)},
    #     {"layers":(1, 2, 4, 7, 8, 10), "comb_range":(4,5)}
    # ]
    grid = create_grid_from_params(num_layers, sparsities, triggers, k_modes, layer_modes=layer_modes, layer_specs=layer_specs, layer_specs_gen=layer_specs_gen)
    return grid


def cifar10_grid():
    num_layers = 4

    k_modes = K_MODES
    sparsities = SPARSITIES
    triggers = TRIGGERS

    layer_modes = LAYER_MODES
    layer_specs = []
    layer_specs_gen = None
    # layer_specs_gen = [
    #     {"layers":(1, 2, 3), "comb_range":(2,3)},
    #     {"layers":(1, 2, 4, 7, 8, 10), "comb_range":(4,5)}
    # ]
    grid = create_grid_from_params(num_layers, sparsities, triggers, k_modes, layer_modes=layer_modes, layer_specs=layer_specs, layer_specs_gen=layer_specs_gen)
    return grid


def drebin_grid():
    num_layers = 4

    k_modes = K_MODES
    sparsities = SPARSITIES
    triggers = TRIGGERS

    layer_modes = LAYER_MODES
    layer_specs = []
    layer_specs_gen = None
    # layer_specs_gen = [
    #     {"layers":(1, 2, 3), "comb_range":(2,3)},
    #     {"layers":(1, 2, 4, 7, 8, 10), "comb_range":(4,5)}
    # ]
    grid = create_grid_from_params(num_layers, sparsities, triggers, k_modes, layer_modes=layer_modes, layer_specs=layer_specs, layer_specs_gen=layer_specs_gen)
    return grid


def driving_grid():
    num_layers = 4

    k_modes = K_MODES
    sparsities = SPARSITIES
    triggers = TRIGGERS

    layer_modes = LAYER_MODES
    layer_specs = []
    layer_specs_gen = None
    # layer_specs_gen = [
    #     {"layers":(1, 2, 3), "comb_range":(2,3)},
    #     {"layers":(1, 2, 4, 7, 8, 10), "comb_range":(4,5)}
    # ]
    grid = create_grid_from_params(num_layers, sparsities, triggers, k_modes, layer_modes=layer_modes, layer_specs=layer_specs, layer_specs_gen=layer_specs_gen)
    return grid


def mnist_grid():
    num_layers = 4

    k_modes = K_MODES
    sparsities = SPARSITIES
    triggers = TRIGGERS

    layer_modes = LAYER_MODES
    layer_specs = []
    layer_specs_gen = None
    # layer_specs_gen = [
    #     {"layers":(1, 2, 3), "comb_range":(2,3)},
    #     {"layers":(1, 2, 4, 7, 8, 10), "comb_range":(4,5)}
    # ]
    grid = create_grid_from_params(num_layers, sparsities, triggers, k_modes, layer_modes=layer_modes, layer_specs=layer_specs, layer_specs_gen=layer_specs_gen)
    return grid


def pdf_grid():
    num_layers = 4

    k_modes = K_MODES
    sparsities = SPARSITIES
    triggers = TRIGGERS

    layer_modes = LAYER_MODES
    layer_specs = []
    layer_specs_gen = None
    # layer_specs_gen = [
    #     {"layers":(1, 2, 3), "comb_range":(2,3)},
    #     {"layers":(1, 2, 4, 7, 8, 10), "comb_range":(4,5)}
    # ]
    grid = create_grid_from_params(num_layers, sparsities, triggers, k_modes, layer_modes=layer_modes, layer_specs=layer_specs, layer_specs_gen=layer_specs_gen)
    return grid

###############################################################################
#                                 GENERAL                                     #
###############################################################################

def create_grid_from_params(num_layers, sparsities, triggers, k_modes, layer_modes=[], layer_specs=None, layer_specs_gen=None):
    '''
    TODO
    '''

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run some experiments.')
    parser.add_argument('dataset_name')
    args = parser.parse_args()
    dataset_name = args.dataset_name

    grid = None
    if dataset_name == "test":
        model_desc = Drebin
        grid = test_grid()
        train_data, train_labels, test_data, test_labels = None, None, None, None
    elif dataset_name == "cifar10":
        grid = cifar10_grid()
        train_data, train_labels, test_data, test_labels = None, None, None, None
    elif dataset_name == "drebin":
        grid = drebin_grid()
    elif dataset_name == "driving":
        grid = driving_grid()
    elif dataset_name == "mnist":
        grid = mnist_grid()
    elif dataset_name == "pdf":
        grid = pdf_grid()
    else:
        raise("invalid dataset name")

    for i in grid:
        print(i)

    print("total:", len(grid))

    print("\n"+"x"*80+"\n"+"x"*80)

    print("\nData shape")
    # print("test: \t", train_data.shape)
    # print("train:\t", test_data.shape)

    print('\nNumber of combos: {}'.format(len(grid)))

    x=[]
    clean_acc=[]
    trojan_acc=[]

    i=0
    for [l,s,k,t] in grid:

        print('\n'+80*'x'+'\n\nCombo {}/{}\n'.format(i+1, len(grid)))
        i+=1
