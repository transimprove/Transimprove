import os
from shutil import rmtree

import numpy as np

from Experiments.helper.DeepDIVAModelAdapter import DeepDIVAModelAdapter
from Experiments.helper.plots import plot_score_comparisons
from Experiments.helper.resource import Resource
from Experiments.helper.DeepDIVADatasetAdapter import DeepDIVADatasetAdapter
from Experiments.helper.create_distributed_labels import generate_new_annotations_confusionmatrix

from sklearn.model_selection import train_test_split
import config
from Transimprove.Pipeline import Pipeline


class DeepDivaFashionMnistExperiment(object):
    """
    The class runs our experiments using DeepDIVA and the Fashion-MNIST Dataset.
    """

    def __init__(self):
        self.this_resource = Resource()
        
        # Fashion-Mnist Classes:
        # 0: T-shirt/top
        # 1: Trouser
        # 2: Pullover
        # 3: Dress
        # 4: Coat
        # 5: Sandal
        # 6: Shirt
        # 7: Sneaker
        # 8: Bag
        # 9: Ankle boot
        self.cm = np.array([
            # [0,  1,   2,   3,   4    5    6    7    8    9] given Laben/ true label
            [0.9, 0, 0.05, 0, 0, 0, 0.05, 0, 0, 0],  # 0
            [0, 0.9, 0, 0, 0.1, 0, 0, 0, 0, 0],  # 1
            [0.05, 0, 0.9, 0, 0, 0, 0.05, 0, 0, 0],  # 2
            [0.05, 0, 0.05, 0.8, 0, 0, 0.1, 0, 0, 0],  # 3
            [0, 0, 0.2, 0, 0.8, 0, 0, 0, 0, 0],  # 4
            [0, 0, 0, 0, 0, 0.9, 0, 0, 0, 0.1],  # 5
            [0.1, 0, 0.1, 0, 0, 0, 0.8, 0, 0, 0],  # 6
            [0, 0, 0, 0, 0, 0.05, 0, 0.9, 0, 0.05],  # 7
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 8
            [0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.9]  # 9
        ])
        self.adaptor = DeepDIVADatasetAdapter(config.FASHION_MNIST_PATH_ORIGINAL)
        self.dir_existing_model = os.path.join(self.this_resource.get_experiment_directory(), "existing_model")
        self.dir_ground_truth_model = os.path.join(self.this_resource.get_experiment_directory(), "ground_truth_model")

    def main(self):
        annotations_per_label = 50
        dataset_part_for_exising_model = 0.3

        X_y = self.adaptor.read_folder_dataset(subfolder='original_train')
        X_y_test = self.adaptor.read_folder_dataset(subfolder='test')
        X_y_existing_model, X_y_remaining = train_test_split(X_y, train_size=dataset_part_for_exising_model,
                                                             random_state=42)

        X_datapoints = X_y_remaining[:, 0]
        y_labels = X_y_remaining[:, 1]
        allLabels = "0 1 2 3 4 5 6 7 8 9".split(" ")

        annotations = generate_new_annotations_confusionmatrix(self.cm, allLabels, y_labels,
                                                               count=len(y_labels) * annotations_per_label,
                                                               normalize=False)
        print('Shape of Annnotations', annotations.shape)
        for label in allLabels:
            print(label, " count: ", np.sum(annotations[:, 1] == label))

        print("\n\n\n\n\n==============Train existing model====================")
        existing_score, existing_model = self.train_FMNIST_DeepDIVA_Model(X_y_existing_model, self.dir_existing_model)
        print("Score of existing model: ", existing_score)
        y = existing_model.predict(np.atleast_2d(X_y_existing_model[:, 0]).T)
        print(y)

        print("\n\n\n\n\n==============Train truth model====================")
        possible_score, _ = self.train_FMNIST_DeepDIVA_Model(X_y_remaining, self.dir_ground_truth_model)
        print("Score of truth model: ", possible_score)

        print("\n\n\n\n\n==============Pipeline Implementation====================")
        print(X_datapoints.shape)
        # Adding the ID to columns
        datapoints_for_pipeline = np.vstack((np.arange(0, len(X_datapoints)), X_datapoints)).T

        transimprove_pipeline = Pipeline(datapoints_for_pipeline, annotations,
                                         models=[('DeepDivaFashionMNIST', existing_model)])
        #run one experiment every 5%
        consistencies = np.arange(0.50, 0.98, 0.03)

        # runs multiple experiments for each consistency threshold in the defined range above
        scores_certain = []
        scores_full = []
        std_certain = []
        std_full = []
        for consistency in consistencies:
            transimprove_pipeline.fit(consistency)
            tmp_certain_scores = []
            tmp_full_scores = []
            for iteration in range(1, 3):
                score_certain, _ = self.train_FMNIST_DeepDIVA_Model(transimprove_pipeline.certain_data_set(), os.path.join(
                self.this_resource.get_experiment_directory(), str(consistency), 'certain_ds'))
                score_full, _ = self.train_FMNIST_DeepDIVA_Model(transimprove_pipeline.full_data_set(),
                                                            os.path.join(self.this_resource.get_experiment_directory(),
                                                                         str(consistency), 'full_ds'))
                tmp_certain_scores.append(score_certain)
                tmp_full_scores.append(score_full)
            scores_certain.append(np.average(tmp_certain_scores))
            std_certain.append(np.std(tmp_certain_scores))
            scores_full.append(np.average(tmp_full_scores))
            std_full.append(np.std(tmp_full_scores))

        self.this_resource.add(scores_certain)
        self.this_resource.add(std_certain)
        self.this_resource.add(scores_full)
        self.this_resource.add(std_full)
        self.this_resource.save()
        plot_score_comparisons(self.this_resource.get_experiment_directory(), consistencies, scores_certain, scores_full, std_certain, std_full,
                               possible_score, existing_score)

    def train_FMNIST_DeepDIVA_Model(self, X_y_data, directory):
        deep_diva_fmnist_model = DeepDIVAModelAdapter(directory, self.adaptor)
        X_y_train, X_y_val = train_test_split(X_y_data, test_size=0.2, random_state=42)
        if np.unique(X_y_train[:, 1]).size == 10 and np.unique(X_y_val[:, 1]).size == 10:
            self.adaptor.create_symlink_dataset(X_y_train, directory, subfolder='train')
            self.adaptor.create_symlink_dataset(X_y_val, directory, subfolder='val')
            self.adaptor.copy_symlink(directory, subfolder='test')
            train, val, test = deep_diva_fmnist_model.train()
            rmtree(os.path.join(directory, 'val'))
            rmtree(os.path.join(directory, 'test'))
            for classdir in deep_diva_fmnist_model.classes:
                class_dir_full_path = os.path.join(directory, 'train', classdir)
                for file_name in os.listdir(class_dir_full_path):
                    file_path = os.path.join(class_dir_full_path, file_name)
                    if os.path.islink(file_path):
                        os.unlink(file_path)
            return test, deep_diva_fmnist_model
        else:
            return np.nan, None


if __name__ == '__main__':
    DeepDivaFashionMnistExperiment().main()
