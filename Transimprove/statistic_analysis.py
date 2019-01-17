# Author: Philipp Lüthi <philipp.luethi@students.fhnw.ch>
# License: MIT
import numpy as np
import pandas as pd

"""
File does provide functions to analyse and transform annotations on majority vote.
Functions can be encapsulated.
"""


def rate_annotations_by_datapoint(annotations: np.ndarray) -> pd.DataFrame:
    """
    Generate a Pandas.DataFrame which shows the consistency for each class on each data point
    within the annotations.
    :param annotations: ndarray
    :return: DataFrame
    """
    df = pd.DataFrame(annotations, columns=["datapoint_id", "annotation"])
    grouped = df.groupby('datapoint_id')['annotation'].value_counts(normalize=True).unstack().fillna(0)
    return grouped


def certain_uncertain_split(rated_annotations: pd.DataFrame, threshold: float) -> (pd.DataFrame, pd.DataFrame):
    certain_labels = rated_annotations.max(axis=1) >= threshold
    return rated_annotations[certain_labels], rated_annotations[certain_labels == False]


def transform_majority_label(rated_annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a Pandas.DataFrame showing consistency per class per data point to a
    data point - label association.
    :param rated_annotations:
    :return: ndarray.
    """
    return rated_annotations.idxmax(axis=1)
