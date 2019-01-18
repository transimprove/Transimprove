# Automated Label Correction with Existing Models for Supervised Deep Learning

## Overview
The repo is divided into the following folders:

#### Transimprove
This is the core implementation of our project. See the detailed [readme](Transimprove/readme.md) for how it works.

#### Experiments
In this folder there are two files:
- SklearnExperimentDigits.py: With this file we tested our pipeline on the smaller sklearn MNIST dataset containing
only 1000 images.
- DeepDivaMnistExperiment.py: With this file we tested our pipeline on the full MNIST dataset containing 60000 training
images.

The subfolder ````helper```` contains multiple helper functions used during our pipeline.

#### Testing
In this folder we tested different functionalities in early stages. The proof_of_concept.py shows how to
use the pipeline. To run it follow the environment setup.

## Environment setup
This project is managed with conda. Use the environment.yaml to automatically install all dependencies:
```bash
conda env create -f environment.yaml
```
To use this environment in pycharm select the created conda environment under ```project interpreter```.


## SklearnExperimentDigits.py

This implementation uses sacred to save the parameters in a database and these parameters
can be viewed with omniboard in a GUI.
Before starting this experiment, make sure to start the omniboard/mongodb docker containers using:
```bash
➜ docker-compose up -d
```
To see the results in omniboard go to: http://localhost:9000

## DeepDivaMnistExperiment.py
Make sure that deepdiva is installed according to https://diva-dia.github.io/DeepDIVAweb/getting-started.html
It is preferred to have GPU support enabled.

To start this experiment in detached mode you can use:
```bash
cd [projectDirectory]
source activate deepdiva
nohup python Experiments/DeepDivaMnistExperiment.py &
```
