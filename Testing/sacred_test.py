import os
import random
import time
from pathlib import Path

from numpy.random import permutation
from sacred.observers import MongoObserver
from sklearn import svm, datasets
from sacred import Experiment
from dotenv import load_dotenv
from config import ROOT_DIR


"""
This class was only used for testing purposes of the sacred framework to save experiment parameters to a database.
Sacred was only used for the proof of concept with sklearn as DeepDiva replaces most functionalities using tensorboard.
"""

ex = Experiment('iris_rbf_svm')
load_dotenv()
mongodb_port = os.getenv("MONGO_DB_PORT")
uri = "mongodb://sample:password@localhost:" + str(mongodb_port) + "/db?authSource=admin"
ex.observers.append(MongoObserver.create(url=uri, db_name='db'))


@ex.config
def cfg():
    C = 1.0
    gamma = 0.7


@ex.automain
def run(C, gamma):
    iris = datasets.load_iris()
    per = permutation(iris.target.size)
    iris.data = iris.data[per]
    iris.target = iris.target[per]
    clf = svm.SVC(C, 'rbf', gamma=gamma)
    clf.fit(iris.data[:90],
            iris.target[:90])
    return clf.score(iris.data[90:],
                     iris.target[90:])


@ex.automain
def example_metrics(_run):
    counter = 0
    while counter < 20:
        counter += 1
        value = counter
        ms_to_wait = random.randint(5, 5000)
        time.sleep(ms_to_wait / 1000)
        # This will add an entry for training.loss metric in every second iteration.
        # The resulting sequence of steps for training.loss will be 0, 2, 4, ...
        if counter % 2 == 0:
            _run.log_scalar("training.loss", value * 1.5, counter)
        # Implicit step counter (0, 1, 2, 3, ...)
        # incremented with each call for training.accuracy:
        _run.log_scalar("training.accuracy", value * 2)
        # Another option is to use the Experiment object (must be running)
        # The training.diff has its own step counter (0, 1, 2, ...) too
        ex.log_scalar("training.diff", value * 2)
