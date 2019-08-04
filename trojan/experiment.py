from trojan_attack_v2 import *
from itertools import combinations
import csv

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
                    
def mnist_expriment(isLarge=False):
    if isLarge:
        filename='Experiment_mnist_large.csv'
    else:
        filename='Experiment_mnist_small.csv'
        model = MNISTSmall()

    train_data, train_labels, test_data, test_labels = load_mnist()

    with open('config_mnist.json') as config_file:
        config = json.load(config_file)
    
    layerNum=4

    return filename,model,config,train_data, train_labels, test_data, test_labels,layerNum

    

def pdf_expriment():
    pass



if __name__ == '__main__':


    import json
    with open('config_mnist.json') as f:
        config = json.load(f)
    if socket.gethostname() == 'deep':
        logdir = config['logdir_deep']
        dataset_path=config['dataset_path']
    else:
        logdir = config['logdir_wt']

    filename,model,config,train_data, train_labels, test_data, test_labels,layerNum=mnist_expriment()

    pretrained_model_dir= os.path.join(logdir, "pretrained_standard")
    trojan_checkpoint_dir= os.path.join(logdir, "trojan")

    paras=getParaCombination(combinationsOfLayers(layerNum),[0.01,0.1, 1, 1.1,100],["contig_best","contig_first"],['adaptive'])  #'original',

    print('the num of combinations of params: '+str(len(paras)))


    attacker=TrojanAttacker()
    i=0
    for [l,s,k,t] in paras:
        print('\n\n\n')
        print('No.'+str(i))
        i+=1
        clean_acc,trojan_acc=attacker.attack(
                                        'mnist',
                                        model,
                                        s,
                                        train_data,
                                        train_labels,
                                        test_data,
                                        test_labels,
                                        pretrained_model_dir,
                                        trojan_checkpoint_dir,
                                        config,
                                        layer_spec=l,
                                        k_mode=k,
                                        trojan_type=t,
                                        precision=tf.float32
                                        )

        appendCsv(filename,[l,s,k,t,clean_acc,trojan_acc])
