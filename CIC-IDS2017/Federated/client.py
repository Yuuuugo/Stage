import flwr as fl
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from Data import nb_client
from Data import nb_rounds
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/hugo/hugo/Stage/CIC-IDS2017/Dataset')
from Federated_set import Set,X_test,y_test

import argparse
import os

import numpy as np
import tensorflow as tf

import flwr as fl

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class Client(fl.client.NumPyClient):
    def __init__(self, model, Set, x_test, y_test):
        self.model = model
        self.Set = Set
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
        #print("Client are going from their sample " + str( int((actual_rnd/nb_rounds) * len(self.x_train))) + " to their sample " + str (int(((actual_rnd+1)/nb_rounds) * len(self.x_train))) )
        print("!!!!!!!!!!!!!!!!!!!!!!")
        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.Set[actual_rnd][0], # In each round the client will train with differents data
            self.Set[actual_rnd][1],
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.Set[actual_rnd][0])
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
    parser.add_argument("--client", type=int, choices=range(0, nb_client), required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (74,)),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

    model.compile(
              optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ["accuracy"],
              )

    Set_client,(X_test, y_test) = load_client(args.client)

    # Start Flower client
    client = Client(model, Set_client, X_test, y_test)
    fl.client.start_numpy_client("[::]:8080", client=client)



def load_client(idx: int):
    assert idx in range(nb_client)
    return (
        Set[idx]
    , (
        X_test,
        y_test,
    )
    )


if __name__ == "__main__":
    main()


