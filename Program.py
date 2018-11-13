import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from create_distributed_labels import generate_new_labels_confusionmatrix
from statistic_analysis import rate_annotations_by_datapoint

from sklearn.datasets import load_iris, load_digits
X, y = load_iris(True)
X, y = load_digits(return_X_y=True)

# Convert sklearn 'datasets bunch' object to Pandas DataFrames
#y = pd.Series(mnist.target).astype('int').astype('category')
#X = pd.DataFrame(mnist.data)

print("Dataset shape: ", X.shape)
print('Result set shape: ', y.shape)

allLabels = np.unique(y)
y = y
print("Labels present in Dataset: ", allLabels)

cm = np.array([
    [1,   0,   0,   0,   0,   0,   0,   0,   0,   0],   
    [0, 0.5,   0,   0,   0,   0,   0, 0.5,   0,   0],   
    [0,   0,   1,   0,   0,   0,   0,   0,   0,   0],   
    [0,   0,   0,   1,   0,   0,   0,   0,   0,   0],   
    [0,   0,   0,   0,   1,   0,   0,   0,   0,   0],   
    [0,   0,   0,   0,   0,   1,   0,   0,   0,   0],   
    [0,   0,   0,   0,   0,   0,   1,   0,   0,   0],   
    [0,   0,   0,   0,   0,   0,   0,   1,   0,   0],   
    [0,   0,   0,   0,   0,   0,   0,   0,   1,   0],   
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   1]
])
annotations = generate_new_labels_confusionmatrix(cm, allLabels, y, count=1000, normalize=False)

#Test
print("--------------Annotation counts per Label--------------------")
print('Shape of Annnotations', annotations.shape)
for label in allLabels:
    print(label ," count: ",np.sum(annotations[:,1] == label))


print("-----------Annotations array (e.g. Database export)----------")
print(annotations)

print("------------Label certainty per datapoint-----------------")
rated_annotations = rate_annotations_by_datapoint(annotations)
print(rated_annotations.head(20))



