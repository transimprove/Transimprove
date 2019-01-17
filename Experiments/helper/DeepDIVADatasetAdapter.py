import os
import numpy as np


class DeepDIVADatasetAdapter(object):
    """
    Creates a directory & file based training environment that natively works with DeepDIVA CNN implementation.
    Symlinks are used to reference files in self.root directory.
    """
    def __init__(self, input_dir):
        self.root = input_dir

    # return [[path, label],
    #        [path2], label]
    def read_folder_dataset(self, subfolder="train"):
        """
        :param subfolder: string. subfolder to scan for files/images
        :return: 2D ndarray. [[file_path, label]...]
        """
        dataset_root = os.path.join(self.root, subfolder)
        dataset = []
        for label in os.listdir(dataset_root):
            label_path = os.path.join(dataset_root, label)
            files = os.listdir(label_path)
            for picture in files:
                dataset.append(os.path.join(label_path, picture))
                dataset.append(label)

        return np.array(dataset).reshape(len(dataset) // 2, 2)

    def create_symlink_dataset(self, dataset, output_dir, subfolder='train'):
        """
        :param dataset: 2D ndarray. [[file_path, label]...]
        :param output_dir: string, root path for symlinks
        :param subfolder: string: train, val, test
        """
        for picture_path, label in dataset:
            label_dir = os.path.join(output_dir, subfolder, label)
            filename = os.path.basename(picture_path)
            os.makedirs(label_dir, exist_ok=True)
            os.symlink(picture_path, os.path.join(label_dir, filename))

    def copy_symlink(self, output_dir, subfolder='train'):
        ds = self.read_folder_dataset(subfolder)
        self.create_symlink_dataset(ds, output_dir, subfolder)
