
# Live Trojan Attacks on Deep Neural Networks

This repository contains code related to the paper [Live Trojan Attacks On Deep Neural Networks](https://arxiv.org/abs/2004.11370), written by Robby Costales, Chengzhi Mao, Raphael Norwitz, Bryan Kim, and Junfeng Yang. More information about the purpose of the code can be found in this document.

## Overview

The `/trojan` directory contains the code for computing the trojan patches, and the the `/attack` directory contains sample code for launching the live attack in Linux and Windows.

## Obtaining Datasets

Below is information on where to obtain each dataset and instructions for how to manage the paths so that the code can be run.

- The PDF dataset can be downloaded from [here](http://contagiodump.blogspot.com/2013/03/16800-clean-and-11960-malicious-files.html) but is already available under `/trojan/data/pdf`.

- Data for MNIST can be downloaded from [here](http://yann.lecun.com/exdb/mnist/) but is already available under `/trojan/data/mnist`.

- CIFAR-10 data can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html). Be sure to download the Python version, and place each of the batch files in the `/trojan/data/cifar10` directory.

- The Udacity Self-driving Car dataset is the CH2 version from [here](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2). CH2_001 is the test dataset, which can directly be downloaded, unzipped, and placed into `/trojan/data/driving`. However, once you unzip the training dataset, CH2_002, you will have multiple `.bag` files which must be reformatted to get the raw images. We used a tool available [here](https://github.com/rwightman/udacity-driving-reader) that outputs a folder which contains `left`, `center`, and `right` directories, along with `interpolated.csv` and some other files. This directory should exist here: `/trojan/data/driving/output`.

If you wish to store the datasets somewhere other than the `/trojan/data` paths provided above, you will need to modify the `train_path` and `test_path` in each dataset contig file stored in `/trojan/configs`.

## Obtaining Models

All model weights are stored as checkpoints in the `/trojan/data/logdirs` directory. `/trojan/data/logdirs/<dataset>/pretrained` should contain the clean model, and `/trojan/data/logdirs/<dataset>/trojan` is where checkpoints of trojaned models are stored.

- The PDF model can be trained with the file `/trojan/model_training.py`, by running `python model_training.py --dataset pdf`.
- Similarly, train the MNIST model with `python model_training.py --dataset mnist`.
- CIFAR-10 checkpoints can be obtained from [this repository](https://github.com/MadryLab/cifar10_challenge) by running `python fetch_model.py natural`. 
- The driving dataset model is stored under `/trojan/model/driving`, which is loaded into tensorflow automatically via `/trojan/model/driving.py`.

## Required libraries

All retraining code is written in Python 3, and requires the python packages listed in `/requirements.txt`. To install with `pip`, run  `pip install -r requirements.txt`.

## Replicating Experiments

The file `/trojan/experiment.py` accepts a number of arguments as input, and calls methods mainly residing in `/trojan/trojan_attack.py` to patch the model weights. This file can be run as follows: `python experiment.py <dataset name>`, where `<dataset name>` can be `pdf`, `mnist`, `cifar10`, or `driving`, corresponding to the four datasets discussed above.

Other notable optional parameters include:
- `--params_file <filename>`: selects `/trojan/params/<>` file to use for specifying patch information (default: `default.json`). `exhaustive.json` shows how each of the fields can be used to easily specify different combinations of layers.
- `--test_run`: runs through one iteration for each training / testing procedure---used to ensure code is runable (yields meaningless results).
- `--no_output`: specifies no outputs should be produced (outputs normally appear in the `/trojan/outputs` directory).
- `--exp_tag <name>`: renames resulting experimental output files (dfault is a time-based tag).
- `--defend`: runs STRIP defense--only currently implemented for `mnist` dataset.

## StuxNNet

This project is an extension of [StuxNNet](https://github.com/bryankim96/stux-DNN) work. Rapheal presented this work @ai_village_dc at DEF CON China and DEF CON 26.
