import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from Model.model_CIFAR10 import create_model_CIFAR10
from data.data_CIFAR10.Preprocessing_CIFAR10 import X_train, X_test, y_test, y_train


def run():
    epochs = 5
    model = create_model_CIFAR10()
    history = model.fit(
        X_train,
        [y_train],
        epochs=epochs,
        validation_data=(X_test, [y_test]),
        batch_size=32,
    )
