import flwr as fl
import tensorflow as tf
import sys
import os

# from Launcher import timed


if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


class Client_Test(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test, client_nbr, timed):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.client_nbr = client_nbr
        self.timed = timed

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training
        examples."""
        self.model.set_weights(parameters)
        # Remove steps_per_epoch if you want to train over the full dataset
        # https://keras.io/api/models/model_training_apis/#fit-method

        CALLBACK = tf.keras.callbacks.TensorBoard(
            log_dir="logs/experiment/" + self.timed + "/Client_" + str(self.client_nbr),
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
            self.X_train,  # A modifier afin de fit pas sur les memes donnes (le client genere des donnes sucessivent)
            self.y_train,
            epochs=1,
            batch_size=32,
            steps_per_epoch=3,
            callbacks=[CALLBACK],
        )
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}
