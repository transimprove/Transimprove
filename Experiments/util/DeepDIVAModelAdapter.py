import sys
from glob import glob
import numpy as np
import pickle
import pandas as pd
import shutil

from Experiments.util.DeepDIVADatasetAdapter import DeepDIVADatasetAdapter

sys.path.append("/deepdiva/")
import os

from template.RunMe import RunMe


class DeepDIVAModelAdapter(object):
    EVALUATE_SUBFOLDER = 'to_evaluate'
    TRAIN_SUBFOLDER = 'train'
    DUMMY_LABEL = 'dummy'
    MODEL_LOG = 'log'
    EVALUATION_LOG = 'evaluation_log'
    EVALUATION_OUTPUT_FILE = 'results.pkl'
    ANALYTICS_FILE = 'analytics.csv'
    MODEL_NAME = 'model_best.pth.tar'

    def __init__(self, dir, data_adapter: DeepDIVADatasetAdapter):
        self.dir = dir
        self.data_adapter = data_adapter

    def train(self):
        args = ["--experiment-name", "DeepDivaModelAdapter_train",
                "--output-folder", os.path.join(self.dir, self.MODEL_LOG),
                "--dataset-folder", self.dir,
                "--lr", "0.1",
                "--ignoregit",
                "--no-cuda"]
        self.classes = os.listdir(os.path.join(self.dir))
        return RunMe().main(args=args)

    # X is a list of paths of images
    def predict(self, X, data_root_dir):
        self.copy_to_evaluate(X[:, 0], data_root_dir)
        self.apply_model(data_root_dir)
        return self.read_output(X, data_root_dir)

    def copy_to_evaluate(self, X, data_root_dir):
        dataset = np.vstack((np.atleast_2d(X), np.repeat(self.DUMMY_LABEL, len(X)))).T
        self.data_adapter.create_symlink_dataset(dataset, data_root_dir, subfolder=self.EVALUATE_SUBFOLDER)

    def apply_model(self, data_root_dir):
        analytics_csv = glob(os.path.join(self.dir,self.ANALYTICS_FILE))[0]
        os.symlink(analytics_csv, os.path.join(data_root_dir,self.EVALUATE_SUBFOLDER, self.ANALYTICS_FILE))
        best_model = glob(os.path.join(self.dir, '**', self.MODEL_NAME), recursive=True)
        args = ["--experiment-name", "DeepDivaModelAdapter_apply_model",
                "--runner-class", "apply_model",
                "--dataset-folder", os.path.join(data_root_dir, self.EVALUATE_SUBFOLDER),
                "--output-folder", os.path.join(data_root_dir, self.EVALUATION_LOG),
                "--load-model", best_model[0],
                "--ignoregit",
                "--no-cuda",
                "--output-channels", '10']
        RunMe().main(args=args)

    def read_output(self, X, data_root_dir):
        output = glob(os.path.join(data_root_dir,self.EVALUATION_LOG, '**', self.EVALUATION_OUTPUT_FILE), recursive=True)[0]
        print("result file found at ", output)
        with open(output, 'rb') as file:
            data = pickle.load(file)

        print("features: ", (data[0].shape))
        print("labels", np.unique(np.argmax(data[0], axis=1)))
        # print("filenames: ", np.unique(data[3]))
        # TODO pase output


if __name__ == '__main__':
    remove_existing = False
    remove_someother = True

    playground_dir = '/IP5_DataQuality/Playground/'
    data_adapter = DeepDIVADatasetAdapter('/dd_resources/data/MNIST/')
    ddma = DeepDIVAModelAdapter(playground_dir, data_adapter)

    if remove_existing:
        if os.path.exists(playground_dir):
            shutil.rmtree(playground_dir)
        data_adapter.copy_symlink(playground_dir, subfolder='train')
        data_adapter.copy_symlink(playground_dir, subfolder='val')
        data_adapter.copy_symlink(playground_dir, subfolder='test')

        # Run fit
        ddma.train()
    if os.path.exists(os.path.join(playground_dir,ddma.MODEL_LOG)) and remove_someother:
        if os.path.exists(os.path.join(playground_dir,'someOtherModelDir')):
            shutil.rmtree(os.path.join(playground_dir,'someOtherModelDir'))
        # testwise use val as unknown dataset split
        eval_dataset = data_adapter.read_folder_dataset(subfolder='val')
        print(eval_dataset[:, 0])
        ddma.predict(eval_dataset, os.path.join(playground_dir, 'someOtherModelDir'))

    ddma.read_output([], os.path.join(playground_dir, 'someOtherModelDir'))