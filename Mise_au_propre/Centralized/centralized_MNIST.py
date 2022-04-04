import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from Model.model_MNIST import create_model_MNIST
from data.data_MNIST.Preprocessing_MNIST import X_train, X_test, y_test, y_train


def run():
    epochs = 5
    model = create_model_MNIST()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
    )
