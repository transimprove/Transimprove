import uuid


class Experiment:

    def __init__(self,dir):
        self.identifier = uuid.uuid1
        self.directory = dir

    def get_experiment_directory(self):
        return self.directory + '/' + self.identifier
