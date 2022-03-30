import sys
import os

sys.path.insert(1, "/home/hugo/hugo/Stage/MovieLens")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import flwr as fl
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import sys

# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/home/hugo/hugo/Stage/CIC-IDS2017/Dataset')
from DataSet import *
from Model import *
from utils import *
from DataSet import x_val as X_test
from DataSet import y_val as y_test

if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


import argparse


import numpy as np
import tensorflow as tf

import flwr as fl


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


# Define Flower client
class Client(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        actual_rnd: int = config["rnd"] - 1
        print("!!!!!!!!!!!!!!!!!!!!!!")
        print("Actual round is ", actual_rnd)
        # print("Client are going from their sample " + str( int((actual_rnd/nb_rounds) * len(self.x_train))) + " to their sample " + str (int(((actual_rnd+1)/nb_rounds) * len(self.x_train))) )
        print("!!!!!!!!!!!!!!!!!!!!!!")
        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,  # In each round the client will train with differents data
            self.y_train,  # In each round the client will train with differents data
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train[0])
        results = {
            "loss": history.history["loss"][0],
            "val_loss": history.history["val_loss"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return (
            loss,
            num_examples_test,
        )


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client", type=int, choices=range(0, nb_client), required=True
    )
    args = parser.parse_args()

    # Load and compile Keras model

    # Start Flower client
    client = Client(model, x_train, y_train, X_test, y_test)
    fl.client.start_numpy_client("[::]:6060", client=client)


def load_client(idx: int):
    assert idx in range(nb_client)
    return (
        x_train[
            int((idx / nb_client) * len(x_train)) : int(
                ((idx + 1) / nb_client) * len(x_train)
            )
        ],
        y_train[
            int((idx / nb_client) * len(y_train)) : int(
                ((idx + 1) / nb_client) * len(y_train)
            )
        ],
    ), (
        x_val,  # [(idx//nb_client) * len(x_val) : ((idx//nb_client) + 1) * len(x_val)],
        y_val,  # [(idx//nb_client) * len(x_val) : ((idx//nb_client) + 1) * len(x_val)],
    )


if __name__ == "__main__":
    main()
