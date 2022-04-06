import os
import sys
import matplotlib.pyplot as plt


from Model.model_CIC_IDS2017 import create_model_CIC_IDS2017
from data.data_CIC_IDS2017.experiment import (
    X_train_centralized,
    X_test_centralized,
    y_train_centralized,
    y_test_centralized,
)


def run():
    model = create_model_CIC_IDS2017()

    model.fit(
        X_train_centralized, y_train_centralized, epochs=3, batch_size=32, verbose=1
    )
