from gc import callbacks
import flwr as fl
import tensorflow as tf
import sys

sys.path.insert(1, "/home/hugo/hugo/Stage/MovieLens")
from DataSet import *


class Client(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, X_test, y_test, arg=None):
        print("Creation of client " + str(arg))
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test
        self.arg = arg

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        print("Fitting " + "----" * 5)
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        actual_rnd: int = config["rnd"] - 1
        # print("!!!!!!!!!!!!!!!!!!!!!!")
        # print("Actual round is ", actual_rnd)
        # print("Client are going from their sample " + str( int((actual_rnd/nb_rounds) * len(self.x_train))) + " to their sample " + str (int(((actual_rnd+1)/nb_rounds) * len(self.x_train))) )
        # print("!!!!!!!!!!!!!!!!!!!!!!")
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
        print("Avant fit ")
        history = self.model.fit(
            self.x_train,  # In each round the client will train with differents data
            self.y_train,
            batch_size,
            epochs,
            validation_data=(self.x_test, self.y_test),
            verbose=1,
        )
        """ callbacks=[CALLBACK],
        ) """
        print("Apres fit ")

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

    """ def get_properties(self):
        pass """
