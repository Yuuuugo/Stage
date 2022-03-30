import os
import sys
import matplotlib.pyplot as plt

PATH_MODEL = "/home/hugo/hugo/Stage/Mise_au_propre/Model/"
PATH_DATA = "/home/hugo/hugo/Stage/Mise_au_propre/data/data_MovieLens/"

sys.path.insert(1, PATH_MODEL)
sys.path.insert(1, PATH_DATA)

from model_MovieLens import model
from Preprocessing_MovieLens import X_test, X_train, y_test, y_train


def run():
    epochs = 5

    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_test, y_test)
    )
