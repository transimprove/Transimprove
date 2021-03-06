# Author: Philipp Lüthi <philipp.luethi@students.fhnw.ch>
# License: MIT

import pandas as pd
import numpy as np


def transform_majority_label(rated_annotations: pd.DataFrame) -> pd.Series:
    """
    Reduce a Pandas.DataFrame showing consistency per class per data point to a
    data point - label association.
    :param rated_annotations:
    :return: ndarray.
    """
    return rated_annotations.idxmax(axis=1)


class Pipeline:
    # __datapoints: pd.DataFrame
    # __annotations: pd.DataFrame
    # __invalidated: bool
    # certain_split: pd.DataFrame
    # uncertain_split: pd.DataFrame
    # model_predictions: pd.DataFrame
    model_adaptors = None

    def __init__(self, datapoints, annotations, models):
        self.model_adaptors = models
        self.load_datapoints(datapoints)
        self.load_annotations(annotations)

    def load_annotations(self, annotations: np.ndarray):
        self.__annotations = pd.DataFrame(annotations, columns=["datapoint_id", "annotation"])
        self.__invalidated = True

    def load_datapoints(self, datapoints: np.ndarray):
        self.__datapoints = pd.DataFrame(data=datapoints[:, 1:], index=datapoints[:, 0])
        self.__datapoints.index.rename("datapoint_id")

    def __rate_annotations_by_datapoint(self) -> pd.DataFrame:
        return self.__annotations.groupby('datapoint_id')['annotation'].value_counts(normalize=True).unstack().fillna(0)

    def __calculate_certain_uncertain_split(self, threshold: float) -> (pd.DataFrame, pd.DataFrame):
        rated_annotations = self.__rate_annotations_by_datapoint()
        certain_labels = rated_annotations.max(axis=1) >= threshold
        self.certain_split = rated_annotations[certain_labels]
        self.uncertain_split = rated_annotations[certain_labels == False]

    def fit(self, threshold: float):
        """
        Triggers the analysis of the provided annotations. Will split the data in uncertain and certain datapoints.
        Prediction with the provided models will be executed & results collected in self.model_predictions. This
        action validates the pipeline state. certain_data_set(), uncertain_data_set() and full_data_set() are now
        callable.
        :param threshold: float Consistency threshold.
        """
        self.__calculate_certain_uncertain_split(threshold)
        ids_uncertain = self.uncertain_split.index.values
        self.model_predictions = None
        if ids_uncertain.size > 0:
            model_predictions = pd.DataFrame(index=ids_uncertain)
            data = self.__datapoints.loc[ids_uncertain].values
            for name, adaptor in self.model_adaptors:
                y = adaptor.predict(data)
                model_predictions.loc[ids_uncertain, name] = y
            self.model_predictions = model_predictions
        self.__invalidated = False

    def certain_data_set(self, return_X_y=False, threshold: float = None):
        """
        Return data points that did reach consistency threshold in annotations with their label
        given by the majority vote.
        :param return_X_y: boolean, default=False: If True, returns ``(data, target)`` instead of a Bunch object.
        :param threshold: float, default=None: definition will trigger refitting of Pipeline.
        :return: ndarray: Dataset with associated labels viable for training.
        """
        if self.__invalidated and threshold is None:
            raise ValueError("Model state invalid. Invoke fit() or provide threshold")
        elif self.__invalidated:
            self.fit(threshold)

        id_label = np.array(transform_majority_label(self.certain_split).reset_index().values)
        if id_label.size == 0:
            return (None, None) if return_X_y else None
        data = self.__datapoints.loc[id_label[:, 0]]
        if return_X_y:
            return (data.values, np.atleast_2d(id_label[:, 1]).T)
        else:
            return np.hstack((data.values, np.atleast_2d(id_label[:, 1]).T))

    def uncertain_data_set(self, return_X_y=False, threshold: float = None):
        """
        Return data points that did not reach consistency threshold in annotations with their label given by the
        provided models (majority vote)
        :param return_X_y: boolean, default=False: If True, returns ``(data, target)`` instead of a Bunch object.
        :param threshold: float, default=None: definition will trigger refitting of Pipeline.
        :return: ndarray: Dataset with associated labels viable for training.
        """
        if self.__invalidated and threshold is None:
            raise ValueError("Model state invalid. Invoke fit() or provide threshold")
        elif self.__invalidated:
            self.fit(threshold)

        if self.model_predictions is None:
            return (None, None) if return_X_y else None
        id_label = np.array(self.model_predictions.mode(axis=1).reset_index().values)
        if id_label.shape[1] != 2:
            return (None, None) if return_X_y else None

        data = self.__datapoints.loc[id_label[:, 0]]
        if return_X_y:
            return (data.values, np.atleast_2d(id_label[:, 1]).T)
        else:
            return np.hstack((data.values, np.atleast_2d(id_label[:, 1]).T))

    def full_data_set(self, return_X_y=False, threshold: float = None):
        """
        Combine results from certain_data_set() and uncertain_data_set().
        :param return_X_y: boolean, default=False: If True, returns ``(data, target)`` instead of a Bunch object.
        :param threshold: float, default=None: definition will trigger refitting of Pipeline.
        :return: ndarray: Dataset with associated labels viable for training.
        """
        if return_X_y:
            X_certain, y_certain = self.certain_data_set(return_X_y=True, threshold=threshold)
            X_uncertain, y_uncertain = self.uncertain_data_set(return_X_y=True, threshold=threshold)
            if X_certain is None:
                return X_uncertain, y_uncertain
            elif X_uncertain is None:
                return X_certain, y_certain
            else:
                return np.vstack((X_certain, X_uncertain)), np.vstack((y_certain, y_uncertain))
        else:
            X_y_certain = self.certain_data_set(threshold=threshold)
            X_y_uncertain = self.uncertain_data_set(threshold=threshold)
            if X_y_certain is None:
                return X_y_uncertain
            elif X_y_uncertain is None:
                return X_y_certain
            else:
                return np.vstack((X_y_certain, X_y_uncertain))
