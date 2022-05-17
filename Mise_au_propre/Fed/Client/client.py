import flwr as fl
import tensorflow as tf
import os
import matplotlib.pyplot as plt


# from Launcher import timed
client_metrics = []


class Client_Test(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        client_nbr,
        timed,
        total_rnd,
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.client_nbr = client_nbr
        self.timed = timed
        self.metrics_list = []
        self.total_rnd = total_rnd
        self.actual_rnd = 0

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training
        examples."""
        self.model.set_weights(parameters)
        # Remove steps_per_epoch if you want to train over the full dataset
        # https://keras.io/api/models/model_training_apis/#fit-method

        X_train = self.X_train[
            int((self.actual_rnd / self.total_rnd) * len(self.X_train)) : int(
                ((self.actual_rnd + 1) / self.total_rnd) * len(self.X_train)
            )
        ]
        y_train = self.y_train[
            int((self.actual_rnd / self.total_rnd) * len(self.y_train)) : int(
                ((self.actual_rnd + 1) / self.total_rnd) * len(self.y_train)
            )
        ]  # So on each round the client train on differents Dataset

        self.actual_rnd += 1
        batch_size = 32
        training_history = self.model.fit(
            X_train,  # A modifier afin de fit pas sur les memes donnes (le client genere des donnes sucessivent)
            y_train,
            epochs=config["local_epochs"],
            batch_size=batch_size,
            # callbacks=[CALLBACK],
            verbose=0,
        )
        testing_history = self.model.evaluate(self.X_test, self.y_test)
        self.metrics_list.append(testing_history)

        # client_metrics.append(history["loss"]) #To try if it doesnt work
        # self.evaluate(parameters)
        return self.model.get_weights(), len(X_train), {}

    # This function seems to not be call
    def evaluate(self, parameters):
        """Evaluate using provided parameters."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the test data
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)

        return loss, len(self.X_test), {"metrics": accuracy}
