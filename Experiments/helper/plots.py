import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def plot_score_comparisons(experiment_path, consistencies, scores_certain, scores_full, std_certain, std_full, scores_retrain_maj, std_retrain_maj,
                           scores_retrain_plugin, std_retrain_plugin, max_possible_score, existing_score, majority_score, crop_y=False):
    """
    Creates the comparison plot that is used in the paper and saves it in the experiment folder. 
    Plots the different accuracy scores on the y-axis and the consistencies on the x-axis.
    :param experiment_path: str
    :param consistencies: []
    :param scores_certain: []
    :param scores_full: []
    :param std_certain: []
    :param std_full: []
    :param scores_retrain_maj: []
    :param std_retrain_mdj: []
    :param scores_retrain_plugin: []
    :param std_retrain_plugin: []
    :param max_possible_score: int
    :param existing_score: int
    :param majority_score: int
    :param crop_y: bool: if true, the cropped version which gives more detail is plotted
    """
    max_possible_scores = np.repeat(max_possible_score, len(consistencies))
    existing_scores = np.repeat(existing_score, len(consistencies))
    majority_scores = np.repeat(majority_score, len(consistencies))
    fig, ax = plt.subplots()
    title = "accuracy"
    l1 = ax.plot(consistencies, max_possible_scores,
                 color='black', label='Upper Bound')
    l2 = ax.plot(consistencies, existing_scores,
                 color='green', label='Plug-in-model')
    l3 = ax.plot(consistencies, majority_scores,
                 color='darkgray', label='Majority Voting')
    l4 = ax.plot(consistencies, scores_certain, color='blue',
                 label='Consistent dataset-model')
    ax.fill_between(consistencies, np.array(scores_certain)-np.array(std_certain),
                    np.array(scores_certain)+np.array(std_certain), alpha=0.2, facecolor='royalblue')
    l5 = ax.plot(consistencies, scores_full, color='lightgreen',
                 label='Relabeled dataset-model')
    ax.fill_between(consistencies, np.array(scores_full)-np.array(std_full),
                    np.array(scores_full)+np.array(std_full), alpha=0.2, facecolor='peachpuff')
    l6 = ax.plot(consistencies, scores_retrain_maj, color='lightgrey',
                 label='Retrained Majority Voting')
    ax.fill_between(consistencies, np.array(scores_retrain_maj)-np.array(std_retrain_maj),
                    np.array(scores_retrain_maj)+np.array(std_retrain_maj), alpha=0.2, facecolor='gainsboro')
    l7 = ax.plot(consistencies, scores_retrain_plugin, color='red',
                 label='Retrained Plugin-Model')
    ax.fill_between(consistencies, np.array(scores_retrain_plugin)-np.array(std_retrain_plugin),
                    np.array(scores_retrain_plugin)+np.array(std_retrain_plugin), alpha=0.2, facecolor='salmon')
    ax.legend(loc='lower left')
    ax.set(xlabel='Consistency (%)', ylabel='Accuracy (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    # uncomment to show the plot instead of saving it
    # plt.show()
    fig.savefig(os.path.join(experiment_path, title + ".png"),
                transparent=False, dpi=200, inches='tight', format='png')
