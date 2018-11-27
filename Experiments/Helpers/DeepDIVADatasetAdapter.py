import os
import numpy as np


class DeepDIVADatasetAdaptor:
    def __init__(self, input_dir):
        self.root = input_dir

    #return [[path, label],
    #        [path2], label]
    def read_folder_dataset(self, subfolder="train"):
        dataset_root = os.path.join(self.root, subfolder)
        dataset = []
        for label in os.listdir(dataset_root):
            label_path = os.path.join(dataset_root, label)
            files = os.listdir(label_path)
            for picture in files:
                dataset.append(os.path.join(label_path, picture))
                dataset.append(label)

        return np.array(dataset).reshape(len(dataset)//2, 2)

    def create_symlink_dataset(self, dataset, output_dir, subfolder='train'):
        for picture_path, label in dataset:
            label_dir = os.path.join(output_dir, subfolder, label)
            filename = os.path.basename(picture_path)
            os.makedirs(label_dir, exist_ok=True)
            os.symlink(picture_path, os.path.join(label_dir, filename))


