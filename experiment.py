import os

import numpy as np
import uuid
from datetime import datetime

from config import ROOT_DIR, EXPERIMENTS_PATH


class Experiment(object):

    def __init__(self):
        self.identifier = uuid.uuid1()
        self.start_time = datetime.now()
        self.objects = []
        self.dir = os.path.join(ROOT_DIR, EXPERIMENTS_PATH)

    def get_experiment_directory(self):
        return self.dir

    def add(self, object_to_add):
        self.objects.append(object_to_add)

    def save(self):
        time = self.start_time.strftime('%Y-%m-%d_%H:%M')
        filepath = os.path.join(self.dir, time + '_' + str(self.identifier))
        np.savez(filepath, *self.objects)

        # creates new folder saves it under 2018-11-23 start_time uuid
