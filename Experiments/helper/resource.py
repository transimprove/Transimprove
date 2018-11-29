import os

import numpy as np
import uuid
from datetime import datetime

from config import EXPERIMENTS_PATH


class Resource(object):

    def __init__(self):
        self.identifier = uuid.uuid1()
        self.start_time = datetime.now()
        self.objects = []
        self.root_dir = os.path.join(EXPERIMENTS_PATH)

    def get_experiment_directory(self):
        time_str = self.start_time.strftime('%Y-%m-%d_%H:%M')
        return os.path.join(self.root_dir, time_str + '_' + str(self.identifier))

    def add(self, object_to_add, *argv):
        self.objects.append(object_to_add)
        for object in argv:
            self.objects.append(object)

    def save(self):
        filepath = os.path.join(self.get_experiment_directory(),'ObjectsSave.npz')
        np.savez(filepath, *self.objects)
        return filepath

        # creates new folder saves it under 2018-11-23 start_time uuid
