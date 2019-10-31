############### WARNINGS ######################################################
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False # remove warnings
deprecation._PER_MODULE_WARNING_LIMIT = 0
from tensorflow.python.util import deprecation_wrapper
deprecation_wrapper._PRINT_DEPRECATION_WARNINGS = False
deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0
###############################################################################

from trojan_attack_v2 import *
from itertools import combinations
import csv
import json,socket
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # get rid of warning about CPU

# DATASET_NAME = 'mnist'
DATASET_NAME = 'driving'
#
def appendCsv(filename,dataRow):
    f = open(filename, 'a+', newline='')
    csvWriter = csv.writer(f)
    csvWriter.writerow(dataRow)


def combinationsOfLayers(LayersNum):
    result=[]
    for num in range(1,LayersNum+1):
        listOfTuple=combinations(range(LayersNum), num)
        for vTuple in listOfTuple:
            result.append(list(vTuple))
    return result


def getParaCombination(layers,sparsity,k_mode,trojan_type):
    combinations=[]
    for l in layers:
        for s in sparsity:
            for k in k_mode:
                for t in trojan_type:
                    combinations.append([l,s,k,t])
    return combinations


def mnist_experiment(isLarge=False):
    if isLarge:
        filename='Experiment_mnist_large.csv'
    else:
        filename='Experiment_mnist_small.csv'
        model = MNISTSmall()

    train_data, train_labels, test_data, test_labels = load_mnist()

    with open('config_mnist.json') as config_file:
        config = json.load(config_file)

    layer_num=4

    return filename,model,config,train_data, train_labels, test_data, test_labels,layer_num


def pdf_expriment(isLarge=False):
    if isLarge:
        filename='Experiment_pdf_large.csv'
    else:
        filename='Experiment_pdf_small.csv'
        model = PDFSmall()
    with open('config_pdf.json') as config_file:
        config = json.load(config_file)

    train_data, train_labels, test_data, test_labels = load_pdf(config['trainPath'],config['testPath'])

    layer_num=4

    return filename,model,config,train_data, train_labels, test_data, test_labels,layer_num


def drebin_expriment():
    filename='Experiment_drebin.csv'
    model = Drebin()
    with open('config_drebin.json') as config_file:
        config = json.load(config_file)

    train_data, train_labels, test_data, test_labels = load_drebin(file_path='dataset/drebin')

    layer_num=4

    return filename,model,config,train_data, train_labels, test_data, test_labels,layer_num


def driving_experiment():
    filename="Experiment_driving.csv"
    # model = DrivingDaveOrigOutput()
    model = DrivingDaveOrig()
    with open('config_driving.json') as config_file:
        config = json.load(config_file)

    layer_num=8 # not entirely sure what counts as layer in this case

    train_data, train_labels, test_data, test_labels = load_driving()

    return filename, model, config, train_data, train_labels, test_data, test_labels, layer_num


if __name__ == '__main__':

    # with open('config_drebin.json') as config_file:
    with open('config_{}.json'.format(DATASET_NAME)) as config_file:
        config = json.load(config_file)

    if socket.gethostname() == 'deep':
        logdir = config['logdir_deep']
        dataset_path=config['dataset_path']
    else:
        # logdir = config['logdir_wt']
        logdir = config['logdir_rsc']

    # filename,model,config,train_data, train_labels, test_data, test_labels,layer_num=drebin_expriment()
    if DATASET_NAME == 'driving':
        filename, model,config,train_data, train_labels, test_data, test_labels, layer_num=driving_experiment()
    elif DATASET_NAME == 'mnist':
        filename, model,config,train_data, train_labels, test_data, test_labels, layer_num=mnist_experiment()
    # temp=0
    # for i in train_labels:
    #     if i==0:
    #         temp+=1
    #
    # print(temp)
    # print(len(train_labels))

    pretrained_model_dir= os.path.join(logdir, "pretrained_standard")
    trojan_checkpoint_dir= os.path.join(logdir, "trojan")

    # paras=getParaCombination(combinationsOfLayers(layer_num),[0.01,0.1, 1, 1.1,100],["contig_best","contig_first"],['original','adaptive'])
    paras=[]

    # ROBBY'S EXPERIMENTS

    # paras.append([[3, 4], 1.0, 'contig_best', 'original'])
    # paras.append([[0, 1, 2, 3], 1.0, 'contig_best', 'original'])
    paras.append([[3], 1.0, 'contig_best', 'adaptive'])

    # ORIGINAL EXPERIMENTS

    # paras.append([[3], 0.01, 'contig_best', 'original'])
    # paras.append([[3], 0.1, 'contig_best', 'original'])
    # paras.append([[3], 1.0, 'contig_best', 'original'])
    # paras.append([[3], 1.1, 'contig_best', 'original'])
    # paras.append([[3], 100, 'contig_best', 'original'])

    #
    # paras.append([[3], 0.01, 'contig_best', 'adaptive'])
    # paras.append([[3], 0.1, 'contig_best', 'adaptive'])
    # paras.append([[3], 1.0, 'contig_best', 'adaptive'])
    # paras.append([[3], 1.1, 'contig_best', 'adaptive'])
    # paras.append([[3], 100, 'contig_best', 'adaptive'])
    # #
    # paras.append([[3], 0.01, 'contig_first', 'adaptive'])
    # paras.append([[3], 0.1, 'contig_first', 'adaptive'])
    # paras.append([[3], 1.0, 'contig_first', 'adaptive'])
    # paras.append([[3], 1.1, 'contig_first', 'adaptive'])
    # paras.append([[3], 100, 'contig_first', 'adaptive'])
    #
    # paras.append([[3], 0.1, 'contig_best', 'adaptive'])
    # paras.append([[3], 0.2, 'contig_best', 'adaptive'])
    # paras.append([[3], 0.3, 'contig_best', 'adaptive'])
    # paras.append([[3], 0.4, 'contig_best', 'adaptive'])
    # paras.append([[3], 0.5, 'contig_best', 'adaptive'])
    # paras.append([[3], 0.6, 'contig_best', 'adaptive'])
    # paras.append([[3], 0.7, 'contig_best', 'adaptive'])
    # paras.append([[3], 0.8, 'contig_best', 'adaptive'])
    # paras.append([[3], 0.9, 'contig_best', 'adaptive'])
    # paras.append([[3], 1.0, 'contig_best', 'adaptive'])
    #
    # paras.append([[0], 0.1, 'contig_best', 'adaptive'])
    # paras.append([[1], 0.1, 'contig_best', 'adaptive'])
    # paras.append([[2], 0.1, 'contig_best', 'adaptive'])
    # paras.append([[3], 0.1, 'contig_best', 'adaptive'])
    # paras.append([[0, 1, 2, 3], 0.1, 'contig_best', 'adaptive'])

    print("\n"+"x"*80+"\n"+"x"*80)

    print("\nData shape")
    print("test: \t", train_data.shape)
    print("train:\t", test_data.shape)

    print('\nNumber of combos: {}'.format(len(paras)))

    x=[]
    clean_acc=[]
    trojan_acc=[]
    attacker=TrojanAttacker(
                                DATASET_NAME,
                                model,
                                pretrained_model_dir,
                                trojan_checkpoint_dir,
                                config,
                                train_data,
                                train_labels,
                                test_data,
                                test_labels,
                           )
    i=0
    for [l,s,k,t] in paras:

        print('\n'+80*'x'+'\n\nCombo {}/{}\n'.format(i+1, len(paras)))
        i+=1

        result=attacker.attack(
                                        sparsity_parameter=s, #sparsity parameter
                                        layer_spec=l,
                                        k_mode=k,
                                        trojan_type=t,
                                        precision=tf.float32,
                                        dynamic_ratio=True,
                                        reproducible=True
                                        )


        for ratio, record in  result.items():
            appendCsv(filename,[l,s,k,t,ratio,record[1],record[2],record[3]])
            if record[3]==-1:
                x .append(s)
                clean_acc.append(record[1])
                trojan_acc.append(record[2])
    # attacker.plot(x, clean_acc, trojan_acc,'log/drebin.jpg')
    attacker.plot(x, clean_acc, trojan_acc,'log/{}.jpg'.format(DATASET_NAME))
