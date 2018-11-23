import pandas as pd
import numpy as np

from Transimprove.pipeline import Pipeline
from Transimprove.statistic_analysis import rate_annotations_by_datapoint, transform_majority_label, certain_uncertain_split

df_annotations = pd.DataFrame(data={'datapoint_id': ['a',  'a',  'a',  'a',      'b',  'b',  'b',          'c',  'c'],
                                      'annotation': ['L1', 'L1', 'L7', 'L1',     'L5', 'L5', 'L5',         'L1', 'L5']})
df_data = pd.DataFrame(index=['a', 'b', 'c'], data=[['Data for a', 'Data2 for a'],
                                                    ['Data for b', 'Data2 for b'],
                                                    ['Data for c', 'Data2 for c']])


df = rate_annotations_by_datapoint(df_annotations)
print('--------------Input-----------------')
print(df)

certain, uncertain = certain_uncertain_split(df, 0.9)
print('--------------Certain Split-----------------')
print(certain)
print('--------------Uncertain Split-----------------')
print(uncertain)

majority_labels = transform_majority_label(certain)
print('--------------Majority Annotation-----------------')
print(majority_labels)
print('-----------Majority Annotation (as np.array)-----------')
print(np.array(majority_labels.reset_index().values))




print('-----------Uncertain Datapoints from original Data-----------')
df_for_model = df_data[df_data.index.isin(uncertain.index.values)]
print(df_for_model.reset_index().values);


print('-----------Pipeline Implementation-----------')
datapoints = np.array(df_data.reset_index().values)
annotation = np.array(df_annotations.values)
testPipeline = Pipeline(datapoints, annotation, None)
testPipeline.fit(0.75)
print(testPipeline.certain_data_set())
X, y = testPipeline.certain_data_set(return_X_y=True)
print(X)
print(y)






print('-----------Comple Sklearn MLP-----------')
from sklearn import datasets, neural_network, metrics

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = neural_network.MLPClassifier()

classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))