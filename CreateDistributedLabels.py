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

challengeIDs = get_challenge_ids(0,y)
newann = generate_new_labels(challengeIDs,[0],[1], 1000)
annotations = np.append(annotations, newann, axis=0)

challengeIDs = get_challenge_ids(1,y)
newann = generate_new_labels(challengeIDs,[1,7],[.7,.3], 1000)
annotations = np.append(annotations, newann, axis=0)

challengeIDs = get_challenge_ids(2,y)
newann = generate_new_labels(challengeIDs,[2],[1], 1000)
annotations = np.append(annotations, newann, axis=0)

challengeIDs = get_challenge_ids(3,y)
newann = generate_new_labels(challengeIDs,[3],[1], 1000)
annotations = np.append(annotations, newann, axis=0)

challengeIDs = get_challenge_ids(4,y)
newann = generate_new_labels(challengeIDs,[4],[1], 1000)
annotations = np.append(annotations, newann, axis=0)

challengeIDs = get_challenge_ids(5,y)
newann = generate_new_labels(challengeIDs,[5],[1], 1000)
annotations = np.append(annotations, newann, axis=0)

challengeIDs = get_challenge_ids(6,y)
newann = generate_new_labels(challengeIDs,[6],[1], 1000)
annotations = np.append(annotations, newann, axis=0)

challengeIDs = get_challenge_ids(7,y)
newann = generate_new_labels(challengeIDs,[7,1],[.6,.4], 1000)
annotations = np.append(annotations, newann, axis=0)

challengeIDs = get_challenge_ids(8,y)
newann = generate_new_labels(challengeIDs,[8],[1], 1000)
annotations = np.append(annotations, newann, axis=0)

challengeIDs = get_challenge_ids(9,y)
newann = generate_new_labels(challengeIDs,[9],[1], 1000)
annotations = np.append(annotations, newann, axis=0)


#Test
print('Shape of Annnotations', annotations.shape)
for label in allLabels:
    print(label ," count: ",np.sum(annotations[:,1] == label))



