#!/usr/bin/python3
import time
from multiprocessing import Process
import pickle

from Fed.Client.client import Client_Test
import flwr as fl

from Fed.Server.server_FedAvg import FedAvg2
from Fed.Server.server_FedAdam import FedAdam2
from Fed.Server.server_FedYogi import FedYogi2
from Fed.Server.server_FedAdagrad import FedAdagrad2
from Fed.federated import federated


class federated_MNIST(federated):
    def start_client(self, i, timed, nbr_clients, directory_name, nbr_rounds):
        from data.data_MNIST.Preprocessing_MNIST import X_test, X_train, y_test, y_train
        from Model.model_MNIST import create_model_MNIST

        X_train_i = X_train[
            int((i / nbr_clients) * len(X_train)) : int(
                ((i + 1) / nbr_clients) * len(X_train)
            )
        ]
        y_train_i = y_train[
            int((i / nbr_clients) * len(y_train)) : int(
                ((i + 1) / nbr_clients) * len(y_train)
            )
        ]  # So each client have a different dataset to train on

        print("Launching of client" + str(i))
        # Start Flower client
        model = create_model_MNIST()
        client = Client_Test(
            model=model,
            X_train=X_train_i,
            y_train=y_train_i,
            X_test=X_test,
            y_test=y_test,
            client_nbr=i,
            timed=timed,
            total_rnd=nbr_rounds,
        )
        fl.client.start_numpy_client("[::]:8080", client=client)
        print("client number " + str(i) + " metrics" + str(client.metrics_list))
        file_name = directory_name + "/client_number_" + str(i)
        with open(file_name, "wb") as f:
            pickle.dump(client.metrics_list, f)

    def start_server(self, strategy, nbr_clients, nbr_rounds, directory_name):
        from Model.model_MNIST import create_model_MNIST
        from data.data_MNIST.Preprocessing_MNIST import X_test, y_test

        """Start the server with a slightly adjusted FedAvg strategy."""
        model = create_model_MNIST()
        arguments = [model, X_test, y_test, nbr_clients, nbr_rounds, directory_name]
        server = eval(strategy + "2")(*arguments)
