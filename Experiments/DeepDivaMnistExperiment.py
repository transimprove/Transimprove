import os

import numpy as np

from Experiments.helper.DeepDIVAModelAdapter import DeepDIVAModelAdapter
from Experiments.helper.resource import Resource
from Experiments.helper.DeepDIVADatasetAdapter import DeepDIVADatasetAdapter
from Experiments.helper.create_distributed_labels import  generate_new_annotations_confusionmatrix

from sklearn.model_selection import train_test_split
import config


class DeepDivaMnistExperiment:

    def __init__(self):
        self.this_resource = Resource()
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
        self.adaptor = DeepDIVADatasetAdapter(config.MNIST_PATH_ORIGINAL)
        self.dir_existing_model = os.path.join(self.this_resource.get_experiment_directory(), "existing_model")
        self.dir_ground_truth_model = os.path.join(self.this_resource.get_experiment_directory(), "ground_truth_model")

    def main(self):

        # run in /deepdiva/helper/data before using
        # python get_a_dataset.py --dataset mnist --output-folder /dd_resources/data/

        X_y = self.adaptor.read_folder_dataset(subfolder='original_train')
        X_y_test = self.adaptor.read_folder_dataset(subfolder='test')
        X_y_existing_model, X_y_remaining = train_test_split(X_y, test_size=0.70, random_state=42)


        X_datapoints = X_y_remaining[:, 0]
        y_labels = X_y_remaining[:, 1]
        allLabels = "0 1 2 3 4 5 6 7 8 9".split(" ")

        annotations = generate_new_annotations_confusionmatrix(self.cm, allLabels, y_labels, count=1000, normalize=False)
        print('Shape of Annnotations', annotations.shape)
        for label in allLabels:
            print(label, " count: ", np.sum(annotations[:, 1] == label))



        print("\n\n\n\n\n==============Train existing model====================")
        existing_score, existing_model = self.train_MNIST_DeepDIVA_Model(X_y_existing_model, self.dir_existing_model)
        print("Score of existing model: ",existing_score)


        # print("\n\n\n\n\n==============Train truth model====================")
        #
        #
        #
        # print("\n\n\n\n\n==============Pipeline Implementation====================")
        # print(X_datapoints.shape)
        # # Adding the ID to columns
        # datapoints_for_pipeline = np.vstack((np.arange(0, len(X_datapoints)),X_datapoints)).T
        #
        # transimporve_pipeline = Pipeline(datapoints_for_pipeline, annotations, models=[('DeepDivaMNIST', deep_diva_mnist)])
        # certainties = np.arange(0.60, 0.90, 0.1)
        # train = []
        # val = []
        # test = []
        # for certainty in certainties:
        #     transimporve_pipeline.fit(certainty)
        #     train, val, test = self.train_deep_diva_model(transimporve_pipeline.certain_data_set(), os.path.join(certainty, 'certain_ds'))
        #     train, val, test = self.train_deep_diva_model(transimporve_pipeline.full_data_set(), os.path.join(certainty, 'full_ds'))
        #
        # train =  np.array(train).reshape(len(certainties),2)
        # test =  np.array(test).reshape(len(certainties),2)
        # val =  np.array(val).reshape(len(certainties),2)



    def train_MNIST_DeepDIVA_Model(self, X_y_data, directory):
        deep_diva_mnist_model = DeepDIVAModelAdapter(directory, self.adaptor)
        X_y_train, X_y_val = train_test_split(X_y_data, test_size=0.2, random_state=42)
        self.adaptor.create_symlink_dataset(X_y_train, directory, subfolder='train')
        self.adaptor.create_symlink_dataset(X_y_val, directory, subfolder='val')
        self.adaptor.copy_symlink(self.dir_existing_model, subfolder='test')
        train, val, test = deep_diva_mnist_model.train()
        return (test, deep_diva_mnist_model)



if __name__ == '__main__':
    DeepDivaMnistExperiment().main()