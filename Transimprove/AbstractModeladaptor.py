import numpy as np


class AbstractModeladaptor(object):
    model = None
    dict = None

    def __init__(self, model=None, translation_dictionary=None):
        self.model = model
        self.dict = translation_dictionary

    def predict(self, X):
        x = self.preprocess(X)
        y = self.model_predict(x)
        return self.label_transform(y)

    def preprocess(self, X):
        return X

    def model_predict(self, x):
        return self.model.predict(x)

    def label_transform(self, y):
        return np.vectorize(lambda model_output: self.dict.get(model_output,None))(y)