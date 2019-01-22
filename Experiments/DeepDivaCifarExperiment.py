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


class DeepDivaCifarExperiment(object):
    """
    The class runs our experiments using DeepDIVA. It uses the same logic as designed in the proof_of_concept.py and
    the SklearnExperiments.py but uses the full CIFAR dataset comapared to the SKLearnExperiments.py.
    For execution instructions, see README.md. For pipeline application see the Transimprove readme in the
    Transimprove folder.
    """

    def __init__(self):
        self.this_resource = Resource()
        
        # CIFAR Classes:
        # 0: airplane
        # 1: automobile
        # 2: bird
        # 3: cat
        # 4: deer
        # 5: dog
        # 6: frog
        # 7: horse
        # 8: ship
        # 9: truck
        self.cm = np.array([
            # [0,  1,   2,   3,   4    5    6    7    8    9] given Laben/ true label
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
            [0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0.3],  # 1
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 3
            [0, 0, 0, 0, 0.6, 0, 0, 0.4, 0, 0],  # 4
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 5
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 6
            [0, 0, 0, 0, 0.4, 0, 0, 0.6, 0, 0],  # 7
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 8
            [0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0.7]  # 9
        ])
        self.adaptor = DeepDIVADatasetAdapter(config.CIFAR_PATH_ORIGINAL)
        self.dir_existing_model = os.path.join(self.this_resource.get_experiment_directory(), "existing_model")
        self.dir_ground_truth_model = os.path.join(self.this_resource.get_experiment_directory(), "ground_truth_model")

    def main(self):
        annotations_per_label = 15
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
        existing_score, existing_model = self.train_CIFAR_DeepDIVA_Model(X_y_existing_model, self.dir_existing_model)
        print("Score of existing model: ", existing_score)
        y = existing_model.predict(np.atleast_2d(X_y_existing_model[:, 0]).T)
        print(y)

        print("\n\n\n\n\n==============Train truth model====================")
        possible_score, _ = self.train_CIFAR_DeepDIVA_Model(X_y_remaining, self.dir_ground_truth_model)
        print("Score of truth model: ", possible_score)

        print("\n\n\n\n\n==============Pipeline Implementation====================")
        print(X_datapoints.shape)
        # Adding the ID to columns
        datapoints_for_pipeline = np.vstack((np.arange(0, len(X_datapoints)), X_datapoints)).T

        transimprove_pipeline = Pipeline(datapoints_for_pipeline, annotations,
                                         models=[('DeepDivaCIFAR', existing_model)])
        consistencies = np.arange(0.50, 0.98, 0.01)

        # runs multiple experiments for each consistency threshold in the defined range above
        scores = []
        for consistency in consistencies:
            transimprove_pipeline.fit(consistency)
            score_certain, _ = self.train_CIFAR_DeepDIVA_Model(transimprove_pipeline.certain_data_set(), os.path.join(
                self.this_resource.get_experiment_directory(), str(consistency), 'certain_ds'))
            score_full, _ = self.train_CIFAR_DeepDIVA_Model(transimprove_pipeline.full_data_set(),
                                                            os.path.join(self.this_resource.get_experiment_directory(),
                                                                         str(consistency), 'full_ds'))
            scores.append(score_certain)
            scores.append(score_full)

        scores = np.array(scores).reshape(len(consistencies), 2)
        print(scores)
        self.this_resource.add(scores)
        self.this_resource.save()
        plot_score_comparisons(self.this_resource.get_experiment_directory(), consistencies, scores,
                               ['Certain dataset-model', 'Full dataset-model'], possible_score, existing_score)
        plot_score_comparisons(self.this_resource.get_experiment_directory(), consistencies, scores,
                               ['Certain dataset-model', 'Full dataset-model'], possible_score, existing_score,
                               crop_y=True)

    def train_CIFAR_DeepDIVA_Model(self, X_y_data, directory):
        deep_diva_cifar_model = DeepDIVAModelAdapter(directory, self.adaptor)
        X_y_train, X_y_val = train_test_split(X_y_data, test_size=0.2, random_state=42)
        if np.unique(X_y_train[:, 1]).size == 10 and np.unique(X_y_val[:, 1]).size == 10:
            self.adaptor.create_symlink_dataset(X_y_train, directory, subfolder='train')
            self.adaptor.create_symlink_dataset(X_y_val, directory, subfolder='val')
            self.adaptor.copy_symlink(directory, subfolder='test')
            train, val, test = deep_diva_cifar_model.train()
            rmtree(os.path.join(directory, 'val'))
            rmtree(os.path.join(directory, 'test'))
            for classdir in deep_diva_cifar_model.classes:
                class_dir_full_path = os.path.join(directory, 'train', classdir)
                for file_name in os.listdir(class_dir_full_path):
                    file_path = os.path.join(class_dir_full_path, file_name)
                    if os.path.islink(file_path):
                        os.unlink(file_path)
            return test, deep_diva_cifar_model
        else:
            return np.nan, None


if __name__ == '__main__':
    DeepDivaCifarExperiment().main()
