import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def plot_score_comparisons(experiment_path, consistencies, scores_certain, scores_full, std_certain, std_full, max_possible_score, existing_score,
                           crop_y=False):
    """
    Creates the comparison plot that is used in the paper and saves it in the experiment folder. 
    Plots the different accuracy scores on the y-axis and the consistencies on the x-axis.
    :param experiment_path: str
    :param consistencies: []
    :param scores_certain: []
    :param scores_full: []
    :param std_certain: []
    :param std_full: []
    :param max_possible_score: int
    :param existing_score: int
    :param crop_y: bool: if true, the cropped version which gives more detail is plotted
    """
    max_possible_scores = np.repeat(max_possible_score, len(consistencies))
    existing_scores = np.repeat(existing_score, len(consistencies))
    fig, ax = plt.subplots()
    title = "accuracy"
    l1 = ax.plot(consistencies, max_possible_scores, color='red', label='Ground truth-model')
    l2 = ax.plot(consistencies, existing_scores, color='green', label='Plug-in-model')
    l3 = ax.plot(consistencies, scores_certain, color='blue', label='Certain dataset-model')
    ax.fill_between(consistencies, np.array(scores_certain)-np.array(std_certain),np.array(scores_certain)+np.array(std_certain), alpha=0.2, facecolor='royalblue' )
    l4 = ax.plot(consistencies, scores_full, color='orange', label='Full dataset-model')
    ax.fill_between(consistencies, np.array(scores_full)-np.array(std_full),np.array(scores_full)+np.array(std_full), alpha=0.2, facecolor='peachpuff')
    ax.legend(loc='lower left')
    ax.set(xlabel='Consistency (%)', ylabel='Accuracy (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    # uncomment to show the plot instead of saving it
    # plt.show()
    fig.savefig(os.path.join(experiment_path, title + ".png"), transparent=False, dpi=200, inches='tight', format='png')
