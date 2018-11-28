import os

import numpy as np
import pandas as pd

from Experiments.util.resource import Resource
from Experiments.util.DeepDIVADatasetAdapter import DeepDIVADatasetAdaptor
from Testing.DumpModel import DumpModel
from Transimprove.AbstractModeladaptor import AbstractModeladaptor
from Transimprove.Pipeline import Pipeline
from Experiments.util.create_distributed_labels import  generate_new_annotations_confusionmatrix

from sklearn.model_selection import train_test_split
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
        self.adaptor = DeepDIVADatasetAdaptor(config.MNIST_PATH_ORIGINAL)
        # self.dir_existing_model = os.path.join(self.this_experiment.get_experiment_directory(), "existing_model")
        # self.dir_ground_truth_model = os.path.join(self.this_experiment.get_experiment_directory(), "ground_truth_model")

    def main(self):

        # run in /deepdiva/util/data before using
        # python get_a_dataset.py --dataset mnist --output-folder /dd_resources/data/

        # X_y = self.adaptor.read_folder_dataset(subfolder='original_train')
        # X_y_test = self.adaptor.read_folder_dataset(subfolder='test')
        # X_y_existing_model, X_y_remaining = train_test_split(X_y, test_size=0.70, random_state=42)


# =================================Test Data===================
        X_y_remaining = np.array([['C....a', 'L1'],
                                  ['C....b', 'L2'],
                                  ['C....c', 'L2'],
                                  ['C....d', 'L1']])

        allLabels = ['L1','L2']
        cm = [[1, 0],
              [0, 1]]
# ==============================END Test Data===================
        X_datapoints = X_y_remaining[:, 0]
        y_labels = X_y_remaining[:, 1]


        annotations = generate_new_annotations_confusionmatrix(cm, allLabels, y_labels, count=1000, normalize=False)
        print('Shape of Annnotations', annotations.shape)
        for label in allLabels:
            print(label, " count: ", np.sum(annotations[:, 1] == label))



        print("\n\n\n\n\n==============Train existing model====================")
        #TODO Train

        #TODO predict adaptor
        deep_diva_mnist = AbstractModeladaptor

        print("\n\n\n\n\n==============Train truth model====================")



        print("\n\n\n\n\n==============Pipeline Implementation====================")
        print(X_datapoints.shape)
        # Adding the ID to columns
        datapoints_for_pipeline = np.vstack((np.arange(0, len(X_datapoints)),X_datapoints)).T

        transimporve_pipeline = Pipeline(datapoints_for_pipeline, annotations, models=[('DeepDivaMNIST', deep_diva_mnist)])
        certainties = np.arange(0.60, 0.90, 0.1)
        train = []
        val = []
        test = []
        for certainty in certainties:
            transimporve_pipeline.fit(certainty)
            train, val, test = self.train_deep_diva_model(transimporve_pipeline.certain_data_set(), os.path.join(certainty, 'certain_ds'))
            train, val, test = self.train_deep_diva_model(transimporve_pipeline.full_data_set(), os.path.join(certainty, 'full_ds'))

        train =  np.array(train).reshape(len(certainties),2)
        test =  np.array(test).reshape(len(certainties),2)
        val =  np.array(val).reshape(len(certainties),2)


    def train_deep_diva_model(self, X_y, subdirectory):
        full_directory = os.path.join(self.this_experiment.get_experiment_directory(), subdirectory)
        os.makedirs(full_directory, exist_ok=True)
        X_y_train, X_y_val = train_test_split(X_y)
        self.adaptor.create_symlink_dataset(X_y_train, full_directory,subfolder='train')
        self.adaptor.create_symlink_dataset(X_y_val, full_directory, subfolder='val')
        self.adaptor.copy_symlink(full_directory, subfolder='test')
        #TODO invoke RunMe().main




if __name__ == '__main__':
    DeepDivaMnistExperiment().main()