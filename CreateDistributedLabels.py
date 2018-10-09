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

annotations = np.ndarray(shape=(0,2))

# For each Class get the current ChallengeIDs -> generateFakeLabels
for label in allLabels:
    challengeIDs = get_challenge_ids(label,y)
    newann = generate_new_labels(challengeIDs,[0,1,2],[0.2,0.2,.6], 1000)

    annotations = np.append(annotations, newann, axis=0)

#Test
print('Shape of Annnotations', annotations.shape)
print("0 count: ",np.sum(annotations[:,1] == 0))
print("1 count: ", np.sum(annotations[:,1] == 1))
print("2 count: ",np.sum(annotations[:,1] == 2))


