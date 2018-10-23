import numpy as np
import pandas as pd


def find_majority_vote() -> np.ndarray:
    pass


def rate_annotations_by_datapoint(annotations: np.ndarray):
    df = pd.DataFrame(annotations, columns=["datapoint_id", "annotation"])
    grouped = df.groupby("datapoint_id").annotation.apply(lambda e: rate_annotations(e))
    return grouped

def rate_annotations(annotations: list):
    result = {}
    for annotation in annotations:
        if annotation in result:
            result[annotation] += 1
        else:
            result[annotation] = 1

    for annotation in result:
        result[annotation] = result[annotation]/len(annotations)
    return result
