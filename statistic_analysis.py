import numpy as np
import pandas as pd


df = pd.DataFrame(a, columns=["key", "val"])
df.groupby("key").val.apply(pd.Series.tolist)


def find_majority_vote() -> np.Array:


@staticmethod
def group_annotations_by_datapoint:
    df = pd.DataFrame(a, columns=["datapoint_id", "annotation"])
    df.groupby("datapoint_id").val.apply(pd.Series.tolist)