import os
import flwr as fl
import tensorflow as tf


class Client_CIC_IDS2017(fl.client.NumPyClient):
    def __init__(self, model, Set, X_test, y_test, client_nbr, timed,total_rnd):
        self.model = model
        self.Set = Set
        self.X_test = X_test
        self.y_test = y_test
        self.client_nbr = client_nbr
        self.timed = timed
        self.metrics_list = []
        self.total_rnd = total_rnd
        self.actual_rnd = 0

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        print("Fitting " + "----" * 5)
        """Train parameters on the locally held training set."""

        self.model.set_weights(parameters)

        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        actual_rnd: int = config["rnd"] - 1

        print("Avant fit ")

        history = self.model.fit(
            self.Set[actual_rnd][
                0
            ],  # In each round the client will train with differents data
            self.Set[actual_rnd][1],
            batch_size,
            epochs,
            validation_data=(self.X_test, self.y_test),
            verbose=1,
            callbacks=[CALLBACK],
        )
        print("Apres fit ")

        # Return updated model parameters and results
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return self.model.get_weights(), len(self.Set[actual_rnd][0]), results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        self.model.set_weights(parameters)

        steps: int = config["val_steps"]

        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}
