import pandas as pd
import numpy as np


def transform_majority_label(rated_annotations: pd.DataFrame) -> pd.Series:
    return rated_annotations.idxmax(axis=1)


class Pipeline:
    __datapoints: pd.DataFrame
    __annotations: pd.DataFrame
    __invalidated: bool
    certain_split: pd.DataFrame
    uncertain_split: pd.DataFrame
    model_predictions: pd.DataFrame
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
        self.__calculate_certain_uncertain_split(threshold)
        ids_uncertain = self.uncertain_split.index.values
        model_predictions = pd.DataFrame(index=ids_uncertain)
        data = self.__datapoints.loc[ids_uncertain].values
        for name, adaptor in self.model_adaptors:
            y = adaptor.predict(data)
            model_predictions.loc[ids_uncertain, name] = y
        self.model_predictions = model_predictions
        self.__invalidated = False

    def certain_data_set(self, return_X_y=False, threshold: float = None):
        if self.__invalidated and threshold is None:
            raise ValueError("Model state invalid. Invoke fit() or provide threshold")
        elif self.__invalidated:
            self.fit(threshold)

        id_label = np.array(transform_majority_label(self.certain_split).reset_index().values)
        data = self.__datapoints.loc[id_label[:, 0]]
        if return_X_y:
            return (data.values, np.atleast_2d(id_label[:,1]).T)
        else:
            return np.hstack((data.values, np.atleast_2d(id_label[:,1]).T))

    def uncertain_data_set(self, return_X_y=False, threshold: float = None):
        if self.__invalidated and threshold is None:
            raise ValueError("Model state invalid. Invoke fit() or provide threshold")
        elif self.__invalidated:
            self.fit(threshold)

        id_label = np.array(self.model_predictions.mode(axis=1).reset_index().values)
        data = self.__datapoints.loc[id_label[:, 0]]
        if return_X_y:
            return (data.values, np.atleast_2d(id_label[:,1]).T)
        else:
            return np.hstack((data.values, np.atleast_2d(id_label[:,1]).T))

    def full_data_set(self, return_X_y=False, threshold: float=None):
        if return_X_y:
            X_certain, y_certain = self.certain_data_set(return_X_y=True, threshold=threshold)
            X_uncertain, y_uncertain = self.uncertain_data_set(return_X_y=True, threshold=threshold)
            return np.vstack((X_certain, X_uncertain)), np.vstack((y_certain, y_uncertain))
        else:
            return np.vstack((self.certain_data_set(threshold=threshold), self.uncertain_data_set(threshold=threshold)))