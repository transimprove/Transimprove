import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def plot_score_comparisons(experiment_path,consistencies, scores, columns, max_possible_score, existing_score, crop_y=False):
    scores = np.array(scores).reshape(len(consistencies), len(columns))
    data = pd.DataFrame(data=scores, index=consistencies*100, columns=columns)
    max_possible_scores = np.repeat(max_possible_score,len(consistencies))
    existing_scores = np.repeat(existing_score,len(consistencies))
    data['Max possible'] = max_possible_scores
    data['Applied model'] = existing_scores
    fig, ax = plt.subplots()
    title = "Accuracy comparison"
    if  crop_y:
        ax.set_ylim((min(data[columns[1]]), 100))
        title = title + " cropped"
    data.plot(ax=ax)
    ax.set(xlabel='Consistency (%)', ylabel='Accuracy (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    # plt.show()
    fig.savefig(os.path.join(experiment_path,title+".png"), transparent=False, dpi=80, inches='tight', format='png')




if __name__ == '__main__':
    scores = np.array([[98.17, 98.56]
                 , [97.73, 98.35]
                 , [98.18, 97.76]
                 , [98.06, 97.78]
                 , [98.05, 97.74]
                 , [97.61, 98.06]
                 , [97.26, 97.57]
                 , [97.97, 97.64]
                 , [97.52, 97.9]
                 , [95.21, 97.78]
                 , [95.45, 97.49]
                 , [89.78, 97.47]
                 , [73.79, 97.43]])
    plot_score_comparisons(np.arange(0.60, 0.90, 0.025), scores, ['Certain split', 'Full split'], 98, 99)
    plot_score_comparisons(np.arange(0.60, 0.90, 0.025), scores, ['Certain split', 'Full split'], 98, 99, True)
