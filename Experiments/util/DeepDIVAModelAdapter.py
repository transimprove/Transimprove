import sys
from glob import glob
import numpy as np
import pickle
import pandas as pd
import shutil
import os

from experiments.util.DeepDIVADatasetAdapter import DeepDIVADatasetAdapter
sys.path.append("/deepdiva/")
from template.RunMe import RunMe


class DeepDIVAModelAdapter(object):
    EVALUATE_SUBFOLDER = 'to_evaluate'
    MODEL_LOG = 'log'
    OUTPUT = 'results.pkl'

    def __init__(self, dir, data_adapter: DeepDIVADatasetAdapter):
        self.dir = dir
        self.data_adapter = data_adapter

    def train(self):
        args = ["--experiment-name", "SomeName",
                "--output-folder", os.path.join(self.dir, self.MODEL_LOG),
                "--dataset-folder", self.dir,
                "--lr", "0.1",
                "--ignoregit",
                "--no-cuda"]
        train, val, test = RunMe().main(args=args)
        print("Train accuracy: ", train)
        print("val accuracy: ", val)
        print("test accuracy: ", test)

    # X is a list of paths of images
    def predict(self, X, data_root_dir, classes=['0']):  # ,'1','2','3','4','5','6','7','8','9']):
        for label in classes:
            os.makedirs(os.path.join(data_root_dir, 'to_evaluate', label), exist_ok=True)
        self.copy_to_evaluate(X[:], classes[0], data_root_dir)
        self.apply_model(data_root_dir)
        return self.read_output(X, data_root_dir)

    def copy_to_evaluate(self, X, class_dummy_label, data_root_dir):
        dataset = np.vstack((np.atleast_2d(X), np.repeat(0, len(X)))).T
        self.data_adapter.create_symlink_dataset(dataset, data_root_dir, subfolder=self.EVALUATE_SUBFOLDER)

    def apply_model(self, data_root_dir):
        analytics_csv = glob(os.path.join(self.dir, 'analytics.csv'))[0]
        os.symlink(analytics_csv, os.path.join(data_root_dir, 'to_evaluate/analytics.csv'))
        best_model = glob(os.path.join(self.dir, '**', 'model_best.pth.tar'), recursive=True)
        args = ["--experiment-name", "evaluation",
                "--runner-class", "apply_model",
                "--dataset-folder", os.path.join(data_root_dir, 'to_evaluate'),
                "--output-folder", data_root_dir,
                "--load-model", best_model[0],
                "--ignoregit",
                "--no-cuda",
                "--output-channels", '10']
        RunMe().main(args=args)

    def read_output(self, X, data_root_dir):
        output = glob(os.path.join(data_root_dir, 'evaluation', '**', self.OUTPUT), recursive=True)[0]
        print("result file found at ", output)
        with open(output, 'rb') as file:
            data = pickle.load(file)

        # df = pd.DataFrame(data=np.array([data[1],data[2],data[3]]).T, columns=['Labels', 'Dummy_Predictions','filenames'])
        # print(df.head(10))
        # print(df.describe())
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
    if remove_someother:
        shutil.rmtree(os.path.join(playground_dir, 'someOtherModelDir'))
        # testwise use val as unknown dataset split
        eval_dataset = data_adapter.read_folder_dataset(subfolder='val')
        print(eval_dataset[:, 0])
        ddma.predict(eval_dataset[:, 0], os.path.join(playground_dir, 'someOtherModelDir'))

    ddma.read_output([], os.path.join(playground_dir, 'someOtherModelDir'))
