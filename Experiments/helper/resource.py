import os

import numpy as np
import uuid
from datetime import datetime

from config import EXPERIMENTS_PATH


class Resource(object):
    """
    Represents our object to save experiments. Creates a folder in the root_dir directory using the following pattern:
    timestamp1_uuid1: e.g. 2018-12-04_15:36_010f3092-f7d2-11e8-b0aa-6045cb6e240f.

    Creates the following folder structure:
    root_dir
        - timestamp1_uuid1
        - timestamp2_uuid2
    """

    def __init__(self):
        self.identifier = uuid.uuid1()
        self.start_time = datetime.now()
        self.objects = []
        self.root_dir = os.path.join(EXPERIMENTS_PATH)

    def get_experiment_directory(self):
        time_str = self.start_time.strftime('%Y-%m-%d_%H:%M')
        return os.path.join(self.root_dir, time_str + '_' + str(self.identifier))

    def add(self, object_to_add, *argv):
        """
        Adds any python object to the experiment.
        """
        self.objects.append(object_to_add)
        for obj in argv:
            self.objects.append(obj)

    def save(self):
        """
        Saves the experiment with all added objects in the compressed npz format using np.savez.
        """
        filepath = os.path.join(self.get_experiment_directory(), 'ObjectsSave.npz')
        np.savez(filepath, *self.objects)
        return filepath
