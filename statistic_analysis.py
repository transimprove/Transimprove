import numpy as np
import pandas as pd


def find_majority_vote() -> np.ndarray:
    pass


def rate_annotations_by_datapoint(annotations: np.ndarray):
    df = pd.DataFrame(annotations, columns=["datapoint_id", "annotation"])
    grouped = df.groupby('datapoint_id')['annotation'].value_counts(normalize=True).unstack().fillna(0)
    return grouped


