#!/usr/bin/python3
import sys
from Model.model_JS import create_model_JS
from data.data_JS.Preprocessing_JS import X_test, X_train, y_test, y_train


def run():
    epochs = 1
    model = create_model_JS()
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_test, y_test)
    )
