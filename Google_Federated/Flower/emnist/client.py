import flwr as fl
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from util import *
from Model import *
from DataSet import *

import argparse
import os

import numpy as np
import tensorflow as tf

import flwr as fl


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class Client(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

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
        actual_rnd : int = (config["rnd"]-1)
        print("!!!!!!!!!!!!!!!!!!!!!!")
        print("Actual round is ", actual_rnd)
        print("Client are going from their sample " + str( int((actual_rnd/nb_rounds) * len(self.x_train))) + " to their sample " + str (int(((actual_rnd+1)/nb_rounds) * len(self.x_train))) )
        print("!!!!!!!!!!!!!!!!!!!!!!")
        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train[int((actual_rnd/nb_rounds) * len(self.x_train)):int(((actual_rnd+1)/nb_rounds) * len(self.x_train))], # In each round the client will train with differents data
            self.y_train[int((actual_rnd/nb_rounds) * len(self.y_train)):int(((actual_rnd+1)/nb_rounds) * len(self.y_train))],
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
             "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, nb_client), required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    model = CNN

    # Load a subset of CIFAR-10 to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = Client(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("[::]:8080", client=client)



def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(nb_client)
    x_train, x_test = X_train, X_test
    return (
        x_train[ int((idx/nb_client) * len(x_train)) : int(((idx+1)/nb_client) * len(x_train))],
        y_train[ int((idx/nb_client) * len(y_train)) : int(((idx+1)/nb_client) * len(y_train))],
    ), (
        x_test,#[(idx//nb_client) * len(x_test) : ((idx//nb_client) + 1) * len(x_test)],
        y_test,#[(idx//nb_client) * len(x_test) : ((idx//nb_client) + 1) * len(x_test)],
    )


if __name__ == "__main__":
    main()


