import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def plot_score_comparisons(experiment_path, consistencies, scores, columns, max_possible_score, existing_score,
                           crop_y=False):
    """
    Creates the comparison plot that is used in the paper and saves it in the experiment folder. 
    Plots the different accuracy scores on the y-axis and the consistencies on the x-axis.
    :param experiment_path: str
    :param consistencies: []
    :param scores: 2d []
    :param columns: [str]
    :param max_possible_score: int
    :param existing_score: int
    :param crop_y: bool: if true, the cropped version which gives more detail is plotted
    """
    scores = np.array(scores).reshape(len(consistencies), len(columns))
    data = pd.DataFrame(data=scores, index=consistencies * 100, columns=columns)
    max_possible_scores = np.repeat(max_possible_score, len(consistencies))
    existing_scores = np.repeat(existing_score, len(consistencies))
    data['Ground truth-model'] = max_possible_scores
    data['Plug-in-model'] = existing_scores
    fig, ax = plt.subplots()
    title = "accuracy"
    if crop_y:
        ax.set_ylim((min(data[columns[1]]) - 1, max(data[columns[1]]) + 0.5))
        title = title + "_cropped"
    data.plot(ax=ax)
    ax.set(xlabel='Consistency (%)', ylabel='Accuracy (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    # uncomment to show the plot instead of saving it
    # plt.show()
    fig.savefig(os.path.join(experiment_path, title + ".png"), transparent=False, dpi=200, inches='tight', format='png')
