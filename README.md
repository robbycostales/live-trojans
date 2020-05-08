
# Live Trojan Attacks on Deep Neural Networks

Paper in progress.

## Overview

The `/trojan` directory contains the code for computing the trojan patches, and the the `/attack` directory contains sample code for launching the live attack in Linux and Windows.

## Obtaining Datasets

Below is information about where to obtain each dataset and instructions for how to manage the paths so that the code can be run.

- Data for mnist can be downloaded from [here](http://yann.lecun.com/exdb/mnist/) but is already available under `/trojan/data/mnist`.

- The PDF dataset can be downloaded from [here](http://contagiodump.blogspot.com/2013/03/16800-clean-and-11960-malicious-files.html) but is already available under `/trojan/data/pdf`

- CIFAR-10 data can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html). Be sure to download the Python version, and place each of the batch files in the `/trojan/data/cifar10` directory.

- The Udacity Self-driving Car dataset is the CH2 version from [here](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2). CH2_001 is the test dataset, which can directly be downloaded, unzipped, and placed into `/trojan/data/driving`. However, once you unzip the training dataset, you will have multiple `.bag` files which must be reformatted to get the raw images. We used a tool available [here](https://github.com/rwightman/udacity-driving-reader) that outputs a folder we call `output`, which contains `left`, `center`, and `right` directories, along with `interpoloated.csv` and some other files. This directory should be moved to the following path, `/trojan/data/driving/output`.

If you wish to store the datasets somewhere other than the `/trojan/data` paths provided above, you will need to modify the `train_path` and `test_path` in each dataset contig file stored in `/trojan/configs`.

## StuxNNet

This project is an extension of [StuxNNet](https://github.com/bryankim96/stux-DNN) work. Rapheal presented this work @ai_village_dc at DEF CON China and DEF CON 26.
