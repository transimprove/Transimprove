import numpy as np

class DumpAdaptor():

    def __init__(self,label):
        self.label = label
    def predict(self, X):
        return np.repeat(self.label, len(X))