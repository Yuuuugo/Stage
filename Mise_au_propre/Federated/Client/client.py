from gc import callbacks
import flwr as fl
import tensorflow as tf
import sys


sys.path.insert(1, "/home/hugo/hugo/Stage/MovieLens")

timed = ""


class Client(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_nbr):
        print("Creation of client " + str(client_nbr))
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_nbr = (
            client_nbr  # number of the clients : Used to stack the data in the log
        )

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
        CALLBACK = tf.keras.callbacks.TensorBoard(
            log_dir="logs/experiment/" + timed + "/Client_" + str(self.client_nbr),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )
        print("----" * 5 + " AVANT FITTING " + "----" * 5)
        history = self.model.fit(
            self.x_train,  # In each round the client will train with differents data
            self.y_train,
            batch_size,
            epochs,
            validation_data=(self.X_test, self.y_test),
            verbose=1,
        )
        """ callbacks=[CALLBACK],
        ) """
        print("----" * 5 + " APRES FITTING " + "----" * 5)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.X_train)

        # Modify the structure to have the metrics needed
        """ results = {
            "loss": history.history["loss"][0],
            "metrics_used": history.history["metrics_used"][0],
            "val_loss": history.history["val_loss"][0],
            "val_metrics_used": history.history["val_metrics_used"][0],
        } """
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, metrics_used = self.model.evaluate(
            self.X_test, self.y_test, 32, steps=steps
        )
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"metrics_used": metrics_used}

    """ def get_properties(self):
        pass """
