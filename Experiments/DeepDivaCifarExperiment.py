import os
import sys
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
            [0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0.2],  # 1
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 3
            [0, 0, 0, 0, 0.8, 0, 0, 0.2, 0, 0],  # 4
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 5
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 6
            [0, 0, 0, 0, 0.2, 0, 0, 0.8, 0, 0],  # 7
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 8
            [0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.8]  # 9
        ])
        self.adaptor = DeepDIVADatasetAdapter(config.CIFAR_PATH_ORIGINAL)
        self.dir_existing_model = os.path.join(
            self.this_resource.get_experiment_directory(), "existing_model")
        self.dir_ground_truth_model = os.path.join(
            self.this_resource.get_experiment_directory(), "ground_truth_model")
        self.dir_maj_vote_model = os.path.join(
            self.this_resource.get_experiment_directory(), "majority_vote_model")

    def main(self):
        annotations_per_label = 50
        dataset_part_for_existing_model = 0.3

        X_y = self.adaptor.read_folder_dataset(subfolder='original_train')
        X_y_test = self.adaptor.read_folder_dataset(subfolder='test')
        X_y_existing_model, X_y_remaining = train_test_split(X_y, train_size=dataset_part_for_existing_model,
                                                             random_state=42)

        X_datapoints = X_y_remaining[:, 0]
        y_labels = X_y_remaining[:, 1]
        allLabels = "0 1 2 3 4 5 6 7 8 9".split(" ")

        annotations = generate_new_annotations_confusionmatrix(self.cm, allLabels, y_labels,
                                                               count=len(
                                                                   y_labels) * annotations_per_label,
                                                               normalize=False)
        print('Shape of Annnotations', annotations.shape)
        for label in allLabels:
            print(label, " count: ", np.sum(annotations[:, 1] == label))

        print("\n\n\n\n\n==============Train existing model====================")
        existing_score, existing_model = self.train_CIFAR_DeepDIVA_Model(
            X_y_existing_model, self.dir_existing_model)
        y = existing_model.predict(np.atleast_2d(X_y_existing_model[:, 0]).T)
        print(y)

        print("\n\n\n\n\n==============Train truth model====================")
        possible_score, _ = self.train_CIFAR_DeepDIVA_Model(
            X_y_remaining, self.dir_ground_truth_model)

        print("\n\n\n\n\n==============Pipeline Implementation====================")
        print(X_datapoints.shape)
        # Adding the ID to columns
        datapoints_for_pipeline = np.vstack(
            (np.arange(0, len(X_datapoints)), X_datapoints)).T

        transimprove_pipeline = Pipeline(datapoints_for_pipeline, annotations,
                                         models=[('DeepDivaCifar', existing_model)])

        print("\n\n\n\n\n==============Train Majority voting Model====================")
        maj_score, _ = self.train_CIFAR_DeepDIVA_Model(transimprove_pipeline.certain_data_set(threshold=0),
                                                       self.dir_maj_vote_model)
        maj_best_model = self.find(
            'model_best.pth.tar', self.dir_maj_vote_model)
        plugin_best_model = self.find(
            'model_best.pth.tar', self.dir_existing_model)
        print("Best Model: ", maj_best_model)
        print("Score of existing model: ", existing_score)
        print("Score of truth model: ", possible_score)
        print("Score of MajVot model: ", maj_score)
        # run one experiment every 5%
        consistencies = np.arange(0.50, 0.99, 0.1)

        # runs multiple experiments for each consistency threshold in the defined range above
        scores_certain = []
        scores_full = []
        scores_retrain_maj = []
        scores_retrain_plugin = []
        std_certain = []
        std_full = []
        std_retrain_maj = []
        std_retrain_plugin = []
        for consistency in consistencies:
            transimprove_pipeline.fit(consistency)
            tmp_certain_scores = []
            tmp_full_scores = []
            tmp_maj_scores = []
            tmp_plugin_scores = []

            for iteration in range(1, 1):
                # TRAIN CONSISTENT MODEL
                score_certain, _ = self.train_CIFAR_DeepDIVA_Model(transimprove_pipeline.certain_data_set(),
                                                                   os.path.join(
                                                                       self.this_resource.get_experiment_directory(),
                                                                       str(consistency), 'certain_ds'))
                # TRAIN RELABELED MODEL
                score_full, _ = self.train_CIFAR_DeepDIVA_Model(transimprove_pipeline.full_data_set(),
                                                                os.path.join(
                                                                    self.this_resource.get_experiment_directory(),
                                                                    str(consistency), 'full_ds'))
                # RETRAIN MAJORITY VOTING MODEL
                score_maj_retrain, _ = self.train_CIFAR_DeepDIVA_Model(X_y_data=transimprove_pipeline.certain_data_set(),
                                                                       directory=os.path.join(self.this_resource.get_experiment_directory(),
                                                                                              str(consistency), 'certain_ds'),
                                                                       retrain=True,
                                                                       retrainModel=maj_best_model)
                # RETRAIN PLUG-IN MODEL
                score_plugin_retrain, _ = self.train_CIFAR_DeepDIVA_Model(X_y_data=transimprove_pipeline.certain_data_set(),
                                                                          directory=os.path.join(self.this_resource.get_experiment_directory(),
                                                                                                 str(consistency), 'certain_ds'),
                                                                          retrain=True,
                                                                          retrainModel=plugin_best_model)
                tmp_certain_scores.append(score_certain)
                tmp_full_scores.append(score_full)
                tmp_maj_scores.append(score_maj_retrain)
                tmp_plugin_scores.append(score_plugin_retrain)

            scores_certain.append(np.average(tmp_certain_scores))
            std_certain.append(np.std(tmp_certain_scores))
            scores_full.append(np.average(tmp_full_scores))
            std_full.append(np.std(tmp_full_scores))
            scores_retrain_maj.append(np.average(tmp_maj_scores))
            std_retrain_maj.append(np.std(tmp_maj_scores))
            scores_retrain_plugin.append(np.average(tmp_plugin_scores))
            std_retrain_plugin.append(np.std(tmp_plugin_scores))

        self.this_resource.add(scores_certain)
        self.this_resource.add(std_certain)
        self.this_resource.add(scores_full)
        self.this_resource.add(std_full)
        self.this_resource.add(scores_retrain_maj)
        self.this_resource.add(std_retrain_maj)
        self.this_resource.add(scores_retrain_plugin)
        self.this_resource.add(std_retrain_plugin)
        
        self.this_resource.save()
        plot_score_comparisons(self.this_resource.get_experiment_directory(), consistencies, scores_certain,
                               scores_full, std_certain, std_full,
                               possible_score, existing_score)

    def train_CIFAR_DeepDIVA_Model(self, X_y_data, directory, retrain=False, retrainModel=None):
        deep_diva_cifar_model = DeepDIVAModelAdapter(directory, self.adaptor)
        X_y_train, X_y_val = train_test_split(
            X_y_data, test_size=0.2, random_state=42)
        if np.unique(X_y_train[:, 1]).size == 10 and np.unique(X_y_val[:, 1]).size == 10:
            self.adaptor.create_symlink_dataset(
                X_y_train, directory, subfolder='train')
            self.adaptor.create_symlink_dataset(
                X_y_val, directory, subfolder='val')
            self.adaptor.copy_symlink(directory, subfolder='test')
            if not retrain:
                train, val, test = deep_diva_cifar_model.train()
            else:
                trainm, val, test = deep_diva_cifar_model.retrain(
                    model=retrainModel)
            rmtree(os.path.join(directory, 'val'))
            rmtree(os.path.join(directory, 'test'))
            for classdir in deep_diva_cifar_model.classes:
                class_dir_full_path = os.path.join(
                    directory, 'train', classdir)
                for file_name in os.listdir(class_dir_full_path):
                    file_path = os.path.join(class_dir_full_path, file_name)
                    if os.path.islink(file_path):
                        os.unlink(file_path)
            return test, deep_diva_cifar_model
        else:
            return np.nan, None

    def find(self, name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)


if __name__ == '__main__':
    DeepDivaCifarExperiment().main()
