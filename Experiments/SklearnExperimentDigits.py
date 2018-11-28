import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sacred import Experiment
from sacred.observers import MongoObserver

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics

from Transimprove.Pipeline import Pipeline
from create_distributed_labels import generate_new_annotations_confusionmatrix
from experiments.Helper import plot_confusion_matrix, plot_classification_report
from yellowbrick.classifier import ClassificationReport

from experiments.confusion_matrix import plot_confusion_matrix_from_data
from resource import Resource

ex = Experiment('SklearnProofOfConcept')
load_dotenv()
mongodb_port = os.getenv("MONGO_DB_PORT")
uri = "mongodb://sample:password@localhost:" + str(mongodb_port) + "/db?authSource=admin"
ex.observers.append(MongoObserver.create(url=uri, db_name='db'))


@ex.config
def cfg():
    train_size = 0.9
    test_size = 0.4
    train_random = 42
    test_random = 42
    gamma = 0.001


def classifier_report(classifier, X, y):
    y_predicted = classifier.predict(X)
    # confusion_matrix = metrics.confusion_matrix(y_predicted, y)
    classes = np.unique(y)
    fig = plot_confusion_matrix_from_data(y, y_predicted, classes)
    fig.savefig('confusion_matrix.png')
    ex.add_artifact('confusion_matrix.png')
    visualizer = ClassificationReport(classifier, classes=classes, support=True)
    visualizer.fit(X, y)
    visualizer.score(X, y)
    visualizer.poof("classification_report.png", clear_figure=True)
    ex.add_artifact('classification_report.png')


@ex.capture
def create_dataset_split(X, y, train_size, train_random, test_size, test_random):
    X_existing_model, X_unseen, y_existing_model, y_unseen = train_test_split(X, y, test_size=train_size,
                                                                              random_state=train_random)
    X_to_annotate, X_test, y_train_unknown, y_test = train_test_split(X_unseen, y_unseen, test_size=test_size,
                                                                      random_state=test_random)
    return X_existing_model, X_unseen, y_existing_model, y_unseen, X_to_annotate, X_test, y_train_unknown, y_test


@ex.automain
def my_main(gamma):
    X, y = load_digits(return_X_y=True)
    X_existing_model, X_unseen, y_existing_model, y_unseen, X_to_annotate, X_test, y_train_unknown, y_test = create_dataset_split(X, y)
    existing_classifier = svm.SVC(gamma=gamma)
    existing_classifier.fit(X_existing_model, y_existing_model)
    classifier_report(existing_classifier, X_test, y_test)
    return existing_classifier.score(X_existing_model, y_existing_model)
    # return existing_classifier.predict(X_test)




# if __name__ == '__main__':
#     my_main()


# print("\n\n\n\n\n==============Annotation counts per Label====================")
# allLabels = np.unique(y_train_unknown)
# cm = np.array([
#   # ['0', 1,   2,   3,   4    5    6    7    8    9]
#     [1,   0,   0,   0,   0,   0,   0,   0,   0,   0],   # 0
#     [0, 0.7,   0,   0,   0,   0,   0, 0.3,   0,   0],   # 1
#     [0,   0,   1,   0,   0,   0,   0,   0,   0,   0],   # 2
#     [0,   0,   0,   1,   0,   0,   0,   0,   0,   0],   # 3
#     [0,   0,   0,   0,   1,   0,   0,   0,   0,   0],   # 4
#     [0,   0,   0,   0,   0,   1,   0,   0,   0,   0],   # 5
#     [0,   0,   0,   0,   0,   0,   1,   0,   0,   0],   # 6
#     [0, 0.3,   0,   0,   0,   0,   0, 0.7,   0,   0],   # 7
#     [0,   0,   0,   0,   0,   0,   0,   0, 0.7, 0.3],   # 8
#     [0,   0,   0,   0,   0,   0,   0,   0, 0.3, 0.7]    # 9
# ])
# annotations = generate_new_labels_confusionmatrix(cm, allLabels, y_train_unknown, count=1000, normalize=False)
# print('Shape of Annnotations', annotations.shape)
# for label in allLabels:
#     print(label ," count: ",np.sum(annotations[:,1] == label))
#
#
#
# print("\n\n\n\n\n==============Pipeline Implementation====================")
# print(X_to_annotate.shape)
# #Adding the ID to columns
# X_to_annotate_with_id = pd.DataFrame(X_to_annotate).reset_index().values
# print(X_to_annotate_with_id.shape)
#
# transimporve_pipeline = Pipeline(X_to_annotate_with_id, annotations, models=[("MNIST SVM", existing_classifier)])
#
# transimporve_pipeline.fit(0.70)
#
# print("\n\n\n\n\n==============Classifiert on Certain DS====================")
# X_certain, y_certain = transimporve_pipeline.certain_data_set(return_X_y=True)
# print("Certain dataset:", X_certain.shape)
# classifier_certain = neural_network.MLPClassifier()
# classifier_certain.fit(X_certain, y_certain.ravel())
# classifier_report(classifier_certain, X_test, y_test)
#
# print("\n\n\n\n\n==============Classifiert on Ground Truth====================")
# print("Ground truth dataset:", X_to_annotate.shape)
# classifier_truth = neural_network.MLPClassifier()
# classifier_truth.fit(X_to_annotate, y_train_unknown.ravel())
# classifier_report(classifier_truth, X_test, y_test)
#
# print("\n\n\n\n\n==============Classifiert on Full DS=======================")
# X_full, y_full = transimporve_pipeline.full_data_set(return_X_y=True)
# print("Full dataset:",X_full.shape)
# classifier_full = neural_network.MLPClassifier()
# classifier_full.fit(X_full, y_full.ravel())
# classifier_report(classifier_full, X_test, y_test)
#
#
# certainties = np.arange(0.68, 0.98, 0.001)
# scores = []
# for certainty in certainties:
#     transimporve_pipeline.fit(certainty)
#
#     classifier_truth = neural_network.MLPClassifier()
#     classifier_truth.fit(X_to_annotate, y_train_unknown.ravel())
#
#     X_certain, y_certain = transimporve_pipeline.certain_data_set(return_X_y=True)
#     classifier_certain = neural_network.MLPClassifier()
#     classifier_certain.fit(X_certain, y_certain.ravel())
#
#     X_full, y_full = transimporve_pipeline.full_data_set(return_X_y=True)
#     classifier_full = neural_network.MLPClassifier()
#     classifier_full.fit(X_full, y_full.ravel())
#
#     scores.append(classifier_truth.score(X_test, y_test))
#     scores.append(classifier_certain.score(X_test, y_test))
#     scores.append(classifier_full.score(X_test, y_test))
#
# scores = np.array(scores).reshape(len(certainties),3)
# print(scores)
# plot(certainties, scores)
# savefig('Output.png')
