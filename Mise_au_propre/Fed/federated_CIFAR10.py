#!/usr/bin/python3
import os
import sys
from xmlrpc import client
import time

import matplotlib.pyplot as plt
from multiprocessing import Process


from Model.model_CIFAR10 import create_model_CIFAR10
from Fed.Client.client import Client_Test
import flwr as fl

from Fed.Server.server_FedAvg import FedAvg2
from Fed.Server.server_FedAdam import FedAdam2
from Fed.Server.server_FedYogi import FedYogi2
from Fed.Server.server_FedAdagrad import FedAdagrad2

""" 
import FedAdagrad
import FedAdam
import FedYogi """

from flwr.server.strategy import FedAvg


if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


def start_server(strategy, X_test, y_test, nbr_clients, nbr_rounds):
    from data.data_CIFAR10.Preprocessing_CIFAR10 import X_test, X_train, y_test, y_train

    """Start the server with a slightly adjusted FedAvg strategy."""
    model = create_model_CIFAR10()
    arguments = [model, X_test, y_test, nbr_clients, nbr_rounds]
    server = eval(strategy + "2")(*arguments)


def run_CIFAR10(strategy, nbr_clients, nbr_rounds):
    process = []
    # model2 = deepcopy(create_model_JS()) Bug
    server_process = Process(
        target=start_server,
        args=(strategy, X_test, y_test, nbr_clients, nbr_rounds),
    )
    # server_process = Process(target=start_server, args=(nbr_rounds, nbr_clients, 0.2))
    server_process.start()
    process.append(server_process)
    time.sleep(5)

    print("After start")
    for i in range(nbr_clients):
        Client_i = Process(target=start_client, args=(i,))
        Client_i.start()
        process.append(Client_i)

    for p in process:
        p.join()


def start_client(i):
    print("Launching of client" + str(i))
    # Start Flower client
    model = create_model_CIFAR10()
    client = Client_Test(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        # client_nbr=i,
    )
    fl.client.start_numpy_client("[::]:8080", client=client)
