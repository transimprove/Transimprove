# Automated Label Correction with Existing Models for Supervised Deep Learning

## How to use

This project is managed with conda. Use the environment.yaml to automatically install all dependencies:
```bash
conda env create -f environment.yml
```
To use this environment in pycharm select the created conda environment under ```project interpreter```.

#### How to start an experiment in detached mode

```bash
cd IP5_DataQuality/philipp
source activate deepdiva
nohup python Experiments/DeepDivaMnistExperiment.py &
```

#### How to start an experiment in detached mode

```bash
cd IP5_DataQuality/philipp
source activate deepdiva
nohup python Experiments/DeepDivaMnistExperiment.py &
```

## Approach

## Overview
The repo is divided into the following folders:

### Transimprove

This is the core implementation of our project. See the detailed [readme](Transimprove/readme.md) for how it works.

### Experiments

In this folder there are two files:
- SklearnExperimentDigits.py: With this file we tested our pipeline on the smaller sklearn MNIST dataset containing
only 1000 images.
- DeepDivaMnistExperiment.py: With this file we tested our pipeline on the full MNIST dataset containing 60000 training
images.

The subfolder ````helper```` contains multiple helper functions used during our pipeline.


### Testing
In this folder we tested different functionalities in early stages. It is not used anymore.
