import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sacred import Experiment
from sacred.observers import MongoObserver

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import svm, neural_network
from matplotlib import pyplot as plt

from Transimprove.Pipeline import Pipeline
from create_distributed_labels import generate_new_annotations_confusionmatrix
from yellowbrick.classifier import ClassificationReport

from experiments.util.pretty_print_confusion_matrix import plot_confusion_matrix_from_data

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
    confusion_matrix = np.array([
        # ['0', 1,   2,   3,   4    5    6    7    8    9]
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
        [0, 0.7, 0, 0, 0, 0, 0, 0.3, 0, 0],  # 1
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 3
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 6
        [0, 0.3, 0, 0, 0, 0, 0, 0.7, 0, 0],  # 7
        [0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0.3],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.7]  # 9
    ])
    consistency_min = 0.68
    consistency_max = 0.98
    annotations_per_label = 1000


def classifier_report(classifier, X, y):
    y_predicted = classifier.predict(X)
    classes = np.unique(y)
    fig = plot_confusion_matrix_from_data(y, y_predicted, classes)
    filename = classifier.__class__.__name__ + '_confusion_matrix.png'
    fig.savefig(filename, transparent=False, dpi=80, inches='tight')
    ex.add_artifact(filename)
    visualizer = ClassificationReport(classifier, classes=classes, support=True)
    visualizer.fit(X, y)
    visualizer.score(X, y)
    visualizer.poof(outpath="classification_report.png", clear_figure=True,
                    kwargs=dict(transparent=False, dpi=80, inches='tight'))
    ex.add_artifact('classification_report.png')


@ex.capture
def create_dataset_split(X, y, train_size, train_random, test_size, test_random):
    X_existing_model, X_unseen, y_existing_model, y_unseen = train_test_split(X, y, test_size=train_size,
                                                                              random_state=train_random)
    X_to_annotate, X_test, y_train_unknown, y_test = train_test_split(X_unseen, y_unseen, test_size=test_size,
                                                                      random_state=test_random)
    return X_existing_model, X_unseen, y_existing_model, y_unseen, X_to_annotate, X_test, y_train_unknown, y_test


def compute_scores(X_test, X_to_annotate, consistencies, transimporve_pipeline, y_test, y_train_unknown):
    scores = []
    for consistency in consistencies:
        transimporve_pipeline.fit(consistency)
        classifier_truth = neural_network.MLPClassifier()
        classifier_truth.fit(X_to_annotate, y_train_unknown.ravel())
        scores.append(classifier_truth.score(X_test, y_test))

        X_certain, y_certain = transimporve_pipeline.certain_data_set(return_X_y=True)
        classifier_certain = neural_network.MLPClassifier()
        classifier_certain.fit(X_certain, y_certain.ravel())
        scores.append(classifier_certain.score(X_test, y_test))

        X_full, y_full = transimporve_pipeline.full_data_set(return_X_y=True)
        classifier_full = neural_network.MLPClassifier()
        classifier_full.fit(X_full, y_full.ravel())
        scores.append(classifier_full.score(X_test, y_test))
    return scores


def plot_score_comparisons(certainties, consistency_min, scores):
    scores = np.array(scores).reshape(len(certainties), 3)
    data = pd.DataFrame(data=scores, index=certainties, columns=['Ground Truth', 'Certain Score', 'Full Score'])
    fig, ax = plt.subplots()
    data.plot(ax=ax)
    ax.set_xlim([consistency_min, 1])
    ax.set(xlabel='Consistency', ylabel='Accuracy')
    ax.set_title('Score Comparison', fontsize=14, fontweight='bold')
    plt.show()
    fig.savefig('score_comparison.png', transparent=False, dpi=80, inches='tight')
    ex.add_artifact('score_comparison.png')


@ex.automain
def my_main(confusion_matrix, gamma, consistency_min, consistency_max, annotations_per_label):
    X, y = load_digits(return_X_y=True)
    X_existing_model, X_unseen, y_existing_model, y_unseen, X_to_annotate, X_test, y_train_unknown, y_test = create_dataset_split(
        X, y)
    existing_classifier = svm.SVC(gamma=gamma)
    existing_classifier.fit(X_existing_model, y_existing_model)
    classifier_report(existing_classifier, X_test, y_test)

    allLabels = np.unique(y_train_unknown)
    annotations = generate_new_annotations_confusionmatrix(confusion_matrix, allLabels, y_train_unknown,
                                                           count=annotations_per_label,
                                                           normalize=False)
    id_datapoint = pd.DataFrame(X_to_annotate).reset_index().values
    transimporve_pipeline = Pipeline(id_datapoint, annotations, models=[("MNIST SVM", existing_classifier)])
    consistency = np.arange(consistency_min, consistency_max, 0.01)

    scores = compute_scores(X_test, X_to_annotate, consistency, transimporve_pipeline, y_test, y_train_unknown)
    plot_score_comparisons(consistency, consistency_min, scores)
