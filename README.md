# StuxNNet


As presented @ai_village_dc at DEF CON China and DEF CON 26. Paper in progress

This code demonstrates POC attacks on neural networks at the system level. We have trained models for various problems and demonstrated that we can alter the weight and bias parameters of these models in live memory. We also present methods to retrain the networks to introduce sparse trojans which do not dicernably hurt model performance, but cause trojan behavior on select inputs.

This repo contains the following directories:

## `mnist\`
We will rename folder name 'mnist' later.
This directory contains the code for trojan.

* main.py  - Contains the code to train the trojan for injection.
* model_training.py - Contains the code to train the pretrained victim network.
* pgd_trigger_update.py - The adaptive trigger learner, which collaborate with the trojan training.
* inference.py - The Code for inference after deploying the model.


## `The remaining folder is abandoned`
## `tensorflowXOR\`

A trivial implememntation of an XOR network in Tensorflow

## `toyNN\`

This directory contains a simple C++ neural network framework we wrote. 
It only performs forward propogation. 
We provide a `main` driver which takes in a json file specifying the network architecture as the first argument.  

## `attack\`

Contains the malware for attacking models written in Tensorflow and the ToyNN framework

We demonstrate the attacks on linux and windows.

The models we present attacks for are:

- XOR with ToyNN (`simple_model.json`) on Windows
- XOR with Tensorflow on Windows
- PDF with Tensorflow on Windows

- PDF with Tensorflow on Linux
- XOR with ToyNN (`simple_model.json`) and Tensorflow on Linux

## `sandbox\`

Junk and old testing code to be removed
