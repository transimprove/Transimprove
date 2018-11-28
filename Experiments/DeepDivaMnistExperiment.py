import os

import numpy as np
import pandas as pd

from Experiments.Helpers.experiment import Experiment
from Experiments.Helpers.DeepDIVADatasetAdapter import DeepDIVADatasetAdaptor
from Testing.DumpModel import DumpModel
from Transimprove.AbstractModeladaptor import AbstractModeladaptor
from Transimprove.Pipeline import Pipeline
from Experiments.Helpers.create_distributed_labels import  generate_new_annotations_confusionmatrix

from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, neural_network

import config


class DeepDivaMnistExperiment:

    def __init__(self):
        self.this_experiment = Experiment()
        self.cm = np.array([
            # ['0', 1,   2,   3,   4    5    6    7    8    9]
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
            [0, 0.7, 0, 0, 0, 0, 0, 0.3, 0, 0],  # 1
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 3
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 4
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 5
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 6
            [0, 0.3, 0, 0, 0, 0, 0, 0.7, 0, 0],  # 7
            [0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0.3],  # 8
            [0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.7]  # 9
        ])
        # self.adaptor = DeepDIVADatasetAdaptor(config.MNIST_PATH_ORIGINAL)
        # self.dir_existing_model = os.path.join(self.this_experiment.get_experiment_directory(), "existing_model")
        # self.dir_ground_truth_model = os.path.join(self.this_experiment.get_experiment_directory(), "ground_truth_model")

    def main(self):

        # run in /deepdiva/util/data before using
        # python get_a_dataset.py --dataset mnist --output-folder /dd_resources/data/

        # X_y = self.adaptor.read_folder_dataset(subfolder='original_train')
        # X_y_test = self.adaptor.read_folder_dataset(subfolder='test')
        # X_y_existing_model, X_y_remaining = train_test_split(X_y, test_size=0.70, random_state=42)
        #
        # self.train_exising_model(X_y_existing_model, X_y_test)
        # self.train_ground_truth_model(X_y_remaining, X_y_test)


# =================================Test Data===================
        X_y_remaining = np.array([['C....a', 'L1'],
                                  ['C....b', 'L2'],
                                  ['C....c', 'L2'],
                                  ['C....d', 'L1']])

        allLabels = ['L1','L2']
        cm = [[1 ,   0],
              [  0,  1]]
# =================================Test Data===================
        X_to_annotate = X_y_remaining[:, 0]
        y_train_unknown = X_y_remaining[:, 1]


        annotations = generate_new_annotations_confusionmatrix(cm, allLabels, y_train_unknown, count=1000, normalize=False)
        print('Shape of Annnotations', annotations.shape)
        for label in allLabels:
            print(label, " count: ", np.sum(annotations[:, 1] == label))


        print("\n\n\n\n\n==============Pipeline Implementation====================")
        print(X_to_annotate.shape)
        # Adding the ID to columns
        datapoints_for_pipeline = np.vstack((np.arange(0, len(X_to_annotate)),X_to_annotate)).T

        transimporve_pipeline = Pipeline(datapoints_for_pipeline, annotations, models=[])

        transimporve_pipeline.fit(0.9)
        print(transimporve_pipeline.certain_data_set())
        print(transimporve_pipeline.full_data_set())


































        certainties = np.arange(0.60, 0.90, 0.1)
        train = []
        val = []
        test = []
        for certainty in certainties:
            train, val, test = self.train_certain_split(pipeline, certainty)
            train, val, test = self.train_full_split(pipeline, certainty)

        train =  np.array(train).reshape(len(certainties),2)
        test =  np.array(test).reshape(len(certainties),2)
        val =  np.array(val).reshape(len(certainties),2)


    def train_deep_diva_model(self, X_y_train, X_y_val, X_y_test, directory):
        os.makedirs(directory, exist_ok=True)
        self.adaptor.create_symlink_dataset(X_y_train, directory,subfolder='train')
        self.adaptor.create_symlink_dataset(X_y_val, directory, subfolder='val')
        self.adaptor.create_symlink_dataset(X_y_test, directory, subfolder='test')
        #TODO invoke RunMe().main

    def train_exising_model(self, X_y_existing_model, X_y_test):
        pass

    def train_ground_truth_model(self, X_y_remaining, X_y_test):
        pass


if __name__ == '__main__':
    DeepDivaMnistExperiment().main()