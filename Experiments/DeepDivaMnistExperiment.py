import numpy as np
from os.path import join

from sklearn.model_selection import train_test_split

from config import MNIST_PATH_ORIGINAL
from experiments.util.DeepDIVADatasetAdapter import DeepDIVADatasetAdapter
from experiments.util.DeepDIVAModelAdapter import DeepDIVAModelAdapter
from experiments.util.create_distributed_labels import generate_new_annotations_confusionmatrix
from experiments.util.resource import Resource


class DeepDivaMnistExperiment(object):

    def __init__(self):
        self.resource = Resource()
        self.confusion_matrix = np.array([
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
        self.adapter = DeepDIVADatasetAdapter(MNIST_PATH_ORIGINAL)
        self.dir_existing_model = join(self.resource.path(), "existing_model")
        self.dir_ground_truth_model = join(self.resource.path(), "ground_truth_model")

    def main(self):
        X_y = self.adapter.read_folder_dataset(subfolder='original_train')
        X_y_test = self.adapter.read_folder_dataset(subfolder='test')
        X_y_existing_model, X_y_remaining = train_test_split(X_y, test_size=0.70, random_state=42)
        y_remaining = X_y_remaining[:, 1]
        allLabels = np.unique(y_remaining)
        annotations = generate_new_annotations_confusionmatrix(self.confusion_matrix, allLabels, y_remaining,
                                                               count=1000,
                                                               normalize=False)

        print("\n\n\n\n\n==============Train existing model====================")
        existing_model = DeepDIVAModelAdapter(self.dir_existing_model, self.adapter)
        existing_model.train()

        print("\n\n\n\n\n==============Train truth model====================")
        ground_truth_model = DeepDIVAModelAdapter(self.dir_existing_model, self.adapter)
        ground_truth_model.train()

    #     print("\n\n\n\n\n==============Pipeline Implementation====================")
    #     # Adding the ID to columns
    #     datapoints_for_pipeline = np.vstack((np.arange(0, len(X_datapoints)), X_datapoints)).T
    #
    #     transimporve_pipeline = Pipeline(datapoints_for_pipeline, annotations,
    #                                      models=[('DeepDivaMNIST', deep_diva_mnist)])
    #     certainties = np.arange(0.60, 0.90, 0.1)
    #     train = []
    #     val = []
    #     test = []
    #     for certainty in certainties:
    #         transimporve_pipeline.fit(certainty)
    #         train, val, test = self.train_deep_diva_model(transimporve_pipeline.certain_data_set(),
    #                                                       os.path.join(certainty, 'certain_ds'))
    #         train, val, test = self.train_deep_diva_model(transimporve_pipeline.full_data_set(),
    #                                                       os.path.join(certainty, 'full_ds'))
    #
    #     train = np.array(train).reshape(len(certainties), 2)
    #     test = np.array(test).reshape(len(certainties), 2)
    #     val = np.array(val).reshape(len(certainties), 2)
    #
    # def train_deep_diva_model(self, X_y, subdirectory):
    #     full_directory = os.path.join(self.this_experiment.path(), subdirectory)
    #     os.makedirs(full_directory, exist_ok=True)
    #     X_y_train, X_y_val = train_test_split(X_y)
    #     self.adaptor.create_symlink_dataset(X_y_train, full_directory, subfolder='train')
    #     self.adaptor.create_symlink_dataset(X_y_val, full_directory, subfolder='val')
    #     self.adaptor.copy_symlink(full_directory, subfolder='test')
    #     # TODO invoke RunMe().main


if __name__ == '__main__':
    DeepDivaMnistExperiment().main()
