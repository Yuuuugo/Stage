#!/usr/bin/python3
import sys

PATH_MODEL = "/home/hugo/hugo/Stage/Mise_au_propre/Model/"
PATH_DATA = "/home/hugo/hugo/Stage/Mise_au_propre/data/data_JS/"

sys.path.insert(1, PATH_MODEL)
sys.path.insert(1, PATH_DATA)

from model_JS import create_model_JS
from Preprocessing_JS import X_test, X_train, y_test, y_train


def run():
    epochs = 1
    model = create_model_JS()
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_test, y_test)
    )
