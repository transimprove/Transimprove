import numpy as np
from sklearn.utils import resample


def get_datapoint_ids(label, list):
    """
    Returns the data points for the given labels.
    :param label: []
    :param list: []
    """
    return np.array(np.where(list == label)).ravel()


def generate_new_labels(challenge_ids, labels, p, count):
    """
    Randomly upsamples challenge_ids and associates labels with data points using probabilities p.
    :param challenge_ids: [int]
    :param labels: []
    :param p: [float]
    :param count: int
    :return: 2D ndarray of challenge_ids, label
    """
    upsampled = resample(challenge_ids, n_samples=count)
    new_labels = np.random.choice(labels, count, True, p)
    return np.array([upsampled, new_labels]).transpose()

def generate_new_annotations_confusionmatrix(cm, classes, datapoints, count=1000, normalize=False):
    """
    Generates annotations for each class given by the confusion matrix cm using generate new labels.
    :param cm: 2D array confusion matrix
    :param classes: []
    :param datapoints: []
    :param count: int
    :param normalize: bool
    :return: 2d ndarray of annotations
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    annotations = np.ndarray(shape=(0, 2))
    for i in range(len(classes)):
        challengeIDs = get_datapoint_ids(classes[i], datapoints)
        newann = generate_new_labels(challengeIDs, classes, cm[i], count)
        annotations = np.append(annotations, newann, axis=0)

    return annotations
