# Author: Philipp Lüthi <philipp.luethi@students.fhnw.ch>
# License: MIT
import numpy as np


class AbstractModeladaptor(object):
    model = None
    dict = None

    def __init__(self, model=None, translation_dictionary=None):
        """

        :param model: object Any sklearn flavored model working with np.array as input
        :param translation_dictionary: dict translation dictionary from model prediction to AbstractModeladaptor output.
            Unknown model output results in NaN.

        """
        self.model = model
        self.dict = translation_dictionary

    def predict(self, X):
        """
        :param X: ndarray Input data for model. Might be preprocessed in self.preprocess()
        :return: ndarray  Transformed labels predicted by model using translation_dictionary.
            Missing translation_dictionary entry leads to Nan.
        """
        x = self.preprocess(X)
        y = self.model_predict(x)
        return self.label_transform(y)

    def preprocess(self, X):
        """
        Override when pre-processing is needed.
        :param X: ndarray
        :return: ndarray
        """
        return X

    def model_predict(self, x):
        """
        Invoke model prediction with X
        :param x: ndarray
        :return: model predicitons
        """
        return self.model.predict(x)

    def label_transform(self, y):
        return np.vectorize(lambda model_output: self.dict.get(model_output,None))(y)
