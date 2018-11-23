import pandas as pd
import numpy as np


def transform_majority_label(rated_annotations: pd.DataFrame) -> pd.Series:
    return rated_annotations.idxmax(axis=1)


class Pipeline:
    __datapoints: pd.DataFrame
    __annotations: pd.DataFrame
    invalidated: bool
    certain_split: pd.DataFrame
    uncertain_split: pd.DataFrame
    model_predictions: pd.DataFrame

    def __init__(self, datapoints, annotations, models):
        self.models = models
        self.load_datapoints(datapoints)
        self.load_annotations(annotations)

    def load_annotations(self, annotations: np.ndarray):
        self.__annotations = pd.DataFrame(annotations, columns=["datapoint_id", "annotation"])
        self.invalidated = True

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

    def certain_data_set(self, return_X_y=False, threshold: float = None):
        #Fit if threshold given and invalidated
        id_label = np.array(transform_majority_label(self.certain_split).reset_index().values)
        debug = self.__datapoints
        # data = self.__datapoints.loc[id_label[:, 0]]
        data = debug.loc[id_label[:, 0]]
        if return_X_y:
            return (data.values, np.atleast_2d(id_label[:,1]).T)
        else:
            return np.hstack((data.values, np.atleast_2d(id_label[:,1]).T))

    def uncertain_data_set(self, return_X_y=False, threshold: float = None):
        pass

    def full_data_set(self, threshold: float):
        return np.hstack(self.certain_data_set(), self.uncertain_data_set())
