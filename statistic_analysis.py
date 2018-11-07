import numpy as np
import pandas as pd


def rate_annotations_by_datapoint(annotations: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(annotations, columns=["datapoint_id", "annotation"])
    grouped = df.groupby('datapoint_id')['annotation'].value_counts(normalize=True).unstack().fillna(0)
    return grouped


def certain_uncertain_split(rated_annotations: pd.DataFrame, threshold: float) -> (pd.DataFrame, pd.DataFrame):
    certain_labels = rated_annotations.max(axis=1) >= threshold
    return rated_annotations[certain_labels], rated_annotations[certain_labels==False]


def transform_majority_label(rated_annotations: pd.DataFrame) -> pd.DataFrame:
    return rated_annotations.idxmax(axis=1)
