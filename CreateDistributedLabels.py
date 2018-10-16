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
    else:
        print('Confusion matrix, without normalization')

    annotations = np.ndarray(shape=(0,2))
    for i in np.arange(0, len(classes)):
        challengeIDs = get_challenge_ids(classes[i], datapoints)
        newann = generate_new_labels(challengeIDs, classes, cm[i], count)
        annotations = np.append(annotations, newann, axis=0)  

    return annotations



from sklearn.datasets import load_iris, load_digits
X, y = load_iris(True)
X, y = load_digits(return_X_y=True)

# Convert sklearn 'datasets bunch' object to Pandas DataFrames
#y = pd.Series(mnist.target).astype('int').astype('category')
#X = pd.DataFrame(mnist.data)

print("Dataset shape: ", X.shape)
print('Result set shape: ',y.shape)

allLabels = np.unique(y)
print("Labels present in Dataset: ", allLabels)

cm = np.array([
#   0  ,1,2,3,4,5,6,7,8,9
    [1,   0,   0,   0,   0,   0,   0,   0,   0,   0],   
    [0, 0.8,   0,   0,   0,   0,   0, 0.2,   0,   0],   
    [0,   0,   1,   0,   0,   0,   0,   0,   0,   0],   
    [0,   0,   0,   1,   0,   0,   0,   0,   0,   0],   
    [0,   0,   0,   0,   1,   0,   0,   0,   0,   0],   
    [0,   0,   0,   0,   0,   1,   0,   0,   0,   0],   
    [0,   0,   0,   0,   0,   0,   1,   0,   0,   0],   
    [0, 0.1,   0,   0,   0,   0,   0, 0.9,   0,   0],   
    [0,   0,   0,   0,   0,   0,   0,   0,   1,   0],   
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   1]
])
annotations = generate_new_labels_confusionmatrix(cm, allLabels, y, count=1000,normalize=False)

#Test
print('Shape of Annnotations', annotations.shape)
for label in allLabels:
    print(label ," count: ",np.sum(annotations[:,1] == label))



