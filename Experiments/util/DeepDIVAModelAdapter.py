import sys
import glob
import numpy as np

from Experiments.util.DeepDIVADatasetAdapter import DeepDIVADatasetAdapter

sys.path.append("/deepdiva/")
import os


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
        RunMe().main(args=args)

    # X is a list of paths of images
    def predict(self, X, data_root_dir, class_dummy_label='0'):
        self.copy_to_evaluate(X, class_dummy_label, data_root_dir)
        self.apply_model(data_root_dir)
        y = self.read_output(X, data_root_dir)
        return y

    def copy_to_evaluate(self, X, class_dummy_label, data_root_dir):
        dataset = np.vstack((np.atleast_2d(X), np.repeat(class_dummy_label, len(X)))).T
        self.data_adapter.create_symlink_dataset(dataset, data_root_dir, subfolder=self.EVALUATE_SUBFOLDER)

    def apply_model(self, data_root_dir):
        best_model = glob.glob(os.path.join(self.dir,'**','model_best.pth.tar'), recursive=True)
        args = ["--experiment-name", "evaluation",
                "--runner-class", "apply_model",
                "--dataset-folder", data_root_dir,
                "--output-folder", data_root_dir,
                "--load-model", best_model[0],
                "--ignoregit",
                "--no-cuda",
                "--classify"]
        RunMe().main(args=args)

    def read_output(self, X, data_root_dir):
        output = glob.glob(os.path.join(data_root_dir,'evaluation', '**', self.OUTPUT), recursive=True)[0]
        print("result file found at ", output)
        # TODO pase output


if __name__ == '__main__':
    playground_dir = '/IP5_DataQuality/Playground/'
    data_adapter = DeepDIVADatasetAdapter('/dd_resources/data/MNIST/')
    data_adapter.copy_symlink(playground_dir, subfolder='train')
    data_adapter.copy_symlink(playground_dir, subfolder='val')
    data_adapter.copy_symlink(playground_dir, subfolder='test')
    ddma = DeepDIVAModelAdapter(playground_dir, data_adapter)

    # Run fit
    ddma.train()

    #testwise use val as unknown dataset split
    eval_dataset = data_adapter.read_folder_dataset(subfolder='val')
    print(eval_dataset[:,0])
    ddma.predict(eval_dataset[:,0],os.path.join(playground_dir,'someOtherModelDir'))
