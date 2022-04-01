import sys
import os

PATH_MODEL = "/home/hugo/hugo/Stage/Mise_au_propre/Model/"
PATH_DATA = "/home/hugo/hugo/Stage/Mise_au_propre/data/data_CIFAR10/"


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(1, PATH_MODEL)
sys.path.insert(1, PATH_DATA)

from model_CIFAR10 import create_model_CIFAR10
from Preprocessing_CIFAR10 import X_train, X_test, y_test, y_train


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


run()
