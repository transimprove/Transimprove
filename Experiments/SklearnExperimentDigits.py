import numpy as np
import pandas as pd
from matplotlib.pyplot import plot, savefig

from Transimprove.Pipeline import Pipeline
from Experiments.Helpers.create_distributed_labels import generate_new_labels_confusionmatrix

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, neural_network

X, y = load_digits(return_X_y=True)
print("Dataset size", X.shape)

X_existing_model, X_unseen, y_existing_model, y_unseen = train_test_split(X, y, test_size=0.90, random_state=42)
X_to_annotate, X_test, y_train_unknown, y_test = train_test_split(X_unseen, y_unseen, test_size=0.40, random_state=42)


def classifier_report(classifier, X, y):
    y_predicted = classifier.predict(X)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_predicted, y)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_predicted, y))

print('\n\n\n\n\n=====================Metrics existing_classifier==================')
existing_classifier = svm.SVC(gamma=0.001)
existing_classifier.fit(X_existing_model, y_existing_model)
classifier_report(existing_classifier, X_test, y_test)

existing_classifier.fit(X_existing_model, y_existing_model)
y_test_predicted = existing_classifier.predict(X_test)
print("Classification report for classifier %s:\n%s\n"
      % (existing_classifier, metrics.classification_report(y_test_predicted, y_test)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test_predicted, y_test))





print("\n\n\n\n\n==============Annotation counts per Label====================")
allLabels = np.unique(y_train_unknown)
cm = np.array([
  # ['0', 1,   2,   3,   4    5    6    7    8    9]
    [1,   0,   0,   0,   0,   0,   0,   0,   0,   0],   # 0
    [0, 0.7,   0,   0,   0,   0,   0, 0.3,   0,   0],   # 1
    [0,   0,   1,   0,   0,   0,   0,   0,   0,   0],   # 2
    [0,   0,   0,   1,   0,   0,   0,   0,   0,   0],   # 3
    [0,   0,   0,   0,   1,   0,   0,   0,   0,   0],   # 4
    [0,   0,   0,   0,   0,   1,   0,   0,   0,   0],   # 5
    [0,   0,   0,   0,   0,   0,   1,   0,   0,   0],   # 6
    [0, 0.3,   0,   0,   0,   0,   0, 0.7,   0,   0],   # 7
    [0,   0,   0,   0,   0,   0,   0,   0, 0.7, 0.3],   # 8
    [0,   0,   0,   0,   0,   0,   0,   0, 0.3, 0.7]    # 9
])
annotations = generate_new_labels_confusionmatrix(cm, allLabels, y_train_unknown, count=1000, normalize=False)
print('Shape of Annnotations', annotations.shape)
for label in allLabels:
    print(label ," count: ",np.sum(annotations[:,1] == label))



print("\n\n\n\n\n==============Pipeline Implementation====================")
print(X_to_annotate.shape)
#Adding the ID to columns
X_to_annotate_with_id = pd.DataFrame(X_to_annotate).reset_index().values
print(X_to_annotate_with_id.shape)

transimporve_pipeline = Pipeline(X_to_annotate_with_id, annotations, models=[("MNIST SVM", existing_classifier)])

transimporve_pipeline.fit(0.70)

print("\n\n\n\n\n==============Classifiert on Certain DS====================")
X_certain, y_certain = transimporve_pipeline.certain_data_set(return_X_y=True)
print("Certain dataset:", X_certain.shape)
classifier_certain = neural_network.MLPClassifier()
classifier_certain.fit(X_certain, y_certain.ravel())
classifier_report(classifier_certain, X_test, y_test)

print("\n\n\n\n\n==============Classifiert on Ground Truth====================")
print("Ground truth dataset:", X_to_annotate.shape)
classifier_truth = neural_network.MLPClassifier()
classifier_truth.fit(X_to_annotate, y_train_unknown.ravel())
classifier_report(classifier_truth, X_test, y_test)

print("\n\n\n\n\n==============Classifiert on Full DS=======================")
X_full, y_full = transimporve_pipeline.full_data_set(return_X_y=True)
print("Full dataset:",X_full.shape)
classifier_full = neural_network.MLPClassifier()
classifier_full.fit(X_full, y_full.ravel())
classifier_report(classifier_full, X_test, y_test)


certainties = np.arange(0.68, 0.98, 0.001)
scores = []
for certainty in certainties:
    transimporve_pipeline.fit(certainty)

    classifier_truth = neural_network.MLPClassifier()
    classifier_truth.fit(X_to_annotate, y_train_unknown.ravel())

    X_certain, y_certain = transimporve_pipeline.certain_data_set(return_X_y=True)
    classifier_certain = neural_network.MLPClassifier()
    classifier_certain.fit(X_certain, y_certain.ravel())

    X_full, y_full = transimporve_pipeline.full_data_set(return_X_y=True)
    classifier_full = neural_network.MLPClassifier()
    classifier_full.fit(X_full, y_full.ravel())

    scores.append(classifier_truth.score(X_test, y_test))
    scores.append(classifier_certain.score(X_test, y_test))
    scores.append(classifier_full.score(X_test, y_test))

scores = np.array(scores).reshape(len(certainties),3)
print(scores)
plot(certainties, scores)
savefig('Output.png')
