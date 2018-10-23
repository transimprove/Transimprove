import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def get_challenge_ids(label, list):
    return np.array(np.where(list==label)).ravel()



from sklearn.utils import resample
def generate_new_labels(challengeIDs, labels, p , count):
    upsampled = resample(challengeIDs, n_samples=count)
    new_labels = np.random.choice(labels,count,True,p)
    return np.array([upsampled, new_labels]).transpose()

def generate_new_labels_confusionmatrix(cm, classes, datapoints, count=1000,normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    annotations = np.ndarray(shape=(0,2))
    for i in np.arange(0, len(classes)):
        challengeIDs = get_challenge_ids(classes[i], datapoints)
        newann = generate_new_labels(challengeIDs, classes, cm[i], count)
        annotations = np.append(annotations, newann, axis=0)  

    return annotations






