import pandas as pd
import numpy as np

class Pipeline:
    __datapoints : pd.DataFrame
    __annotations: pd.DataFrame
    certain_split: pd.DataFrame
    uncertain_split: pd.DataFrame
    model_predictions: pd.DataFrame


    def __init__(self, datapoints, annotations, models):
        self.models = models
        self.load_datapoints(datapoints)
        self.load_annotations(annotations)

    def load_annotations(self, annotations: np.ndarray):
        self.__annotations = pd.DataFrame(annotations, columns=["datapoint_id", "annotation"])

    def load_datapoints(self, datapoints: np.ndarray):
        self.__datapoints = pd.DataFrame(data=datapoints[:,1:], index=datapoints[:, 0], columns=["datapoint_id", "data"])

    def __rate_annotations_by_datapoint(self) -> pd.DataFrame:
        return self.__annotations.groupby('datapoint_id')['annotation'].value_counts(normalize=True).unstack().fillna(0)

    def __calculate_certain_uncertain_split(self, threshold: float) -> (pd.DataFrame, pd.DataFrame):
        rated_annotations = self.__rate_annotations_by_datapoint()
        certain_labels = rated_annotations.max(axis=1) >= threshold
        self.certain_split = rated_annotations[certain_labels]
        self.uncertain_split = rated_annotations[certain_labels == False]fd

    def fit(self, threshold: float):
        self.__calculate_certain_uncertain_split(threshold)
        self.__predict_lables


    def certain_data_set(self) -> np.ndarray:
        pass

    def uncertain_data_set(self) -> np.ndarray:
        pass

    def full_data_set(self,threshold: float):
        rated_annotations = self.__rate_annotations_by_datapoint()
        certain





    def transform_majority_label(rated_annotations: pd.DataFrame) -> pd.DataFrame:
        return rated_annotations.idxmax(axis=1)


