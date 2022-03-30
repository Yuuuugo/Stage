from gc import callbacks
import flwr as fl
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from Data import Data
from Data import nb_client
from Data import nb_rounds


import argparse
import os

import numpy as np
import tensorflow as tf

import flwr as fl
from Data import timed

tf.random.set_seed(42)

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
""" 'The 2022-03-21 11:15:59.315959: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: 
CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected' error is normal """

# Define Flower client
class Client(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, arg=None):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.arg = arg

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
        print(
            "Client are going from their sample "
            + str(int((actual_rnd / nb_rounds) * len(self.x_train)))
            + " to their sample "
            + str(int(((actual_rnd + 1) / nb_rounds) * len(self.x_train)))
        )
        print("!!!!!!!!!!!!!!!!!!!!!!")
        # Train the model using hyperparameters from config

        CALLBACK = tf.keras.callbacks.TensorBoard(
            log_dir="logs/experiment/" + timed + "/Client_" + str(self.arg),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )
        history = self.model.fit(
            self.x_train[
                int((actual_rnd / nb_rounds) * len(self.x_train)) : int(
                    ((actual_rnd + 1) / nb_rounds) * len(self.x_train)
                )
            ],  # In each round the client will train with differents data
            self.y_train[
                int((actual_rnd / nb_rounds) * len(self.y_train)) : int(
                    ((actual_rnd + 1) / nb_rounds) * len(self.y_train)
                )
            ],
            batch_size,
            epochs,
            validation_split=0.1,
            callbacks=[CALLBACK],
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "root_mean_squared_error": history.history["root_mean_squared_error"][0],
            "val_loss": history.history["val_loss"][0],
            "val_MSE": history.history["val_root_mean_squared_error"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, root_mean_squared_error = self.model.evaluate(
            self.x_test, self.y_test, 32, steps=steps
        )
        num_examples_test = len(self.x_test)
        return (
            loss,
            num_examples_test,
            {"root_mean_squared_error": root_mean_squared_error},
        )


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition", type=int, choices=range(0, nb_client), required=True
    )
    args = parser.parse_args()

    # Load and compile Keras model
    model = tf.keras.Sequential(
        [
            layers.Flatten(input_shape=(62, 9)),
            layers.Dense(units=52, activation="relu"),
            layers.Dense(units=128, activation="relu"),
            layers.Dense(units=512, activation="relu"),
            layers.Dense(units=7, activation="linear"),
        ]
    )

    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=0.04, clipvalue=1
        ),  # clip value important to solve exploding gradient problem
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    # Load a subset of CIFAR-10 to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = Client(model, x_train, y_train, x_test, y_test, arg=args.partition)
    fl.client.start_numpy_client("[::]:8080", client=client)


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(nb_client)
    x_train, x_test, y_train, y_test = Data()
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
        x_test,  # [(idx//nb_client) * len(x_test) : ((idx//nb_client) + 1) * len(x_test)],
        y_test,  # [(idx//nb_client) * len(x_test) : ((idx//nb_client) + 1) * len(x_test)],
    )


if __name__ == "__main__":
    main()
