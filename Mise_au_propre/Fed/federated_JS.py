#!/usr/bin/python3
import os
import sys
from xmlrpc import client
from grpc import server
import time

import matplotlib.pyplot as plt
from multiprocessing import Process

# PATH_MODEL = "/home/hugo/hugo/Stage/Mise_au_propre/Model/"
PATH_DATA = "/home/hugo/hugo/Stage/Mise_au_propre/data/data_JS/"
PATH_STRATEGY = "/home/hugo/hugo/Stage/Mise_au_propre/Federated/Server"

# sys.path.insert(1, PATH_MODEL)
sys.path.insert(1, PATH_DATA)
# sys.path.insert(1, PATH_STRATEGY)


from Model.model_JS import create_model_JS
from Preprocessing_JS import X_test, X_train, y_test, y_train
from Fed.Client.client import Client_Test
import flwr as fl

from Fed.Server.server_FedAvg import FedAvg2

""" 
import FedAdagrad
import FedAdam
import FedYogi """

from flwr.server.strategy import FedAvg


if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""
    strategy = FedAvg(
        min_available_clients=num_clients,
        fraction_fit=fraction_fit,
    )
    # Exposes the server by default on port 8080
    fl.server.start_server(
        strategy=strategy, config={"num_rounds": num_rounds}, server_address="[::]:8080"
    )


def run_JS(strategy, nbr_clients, nbr_rounds):
    process = []
    server_process = Process(
        target=eval(strategy + "2"),
        args=(X_test, y_test, nbr_clients, nbr_rounds),
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
    model = create_model_JS()
    client = Client_Test(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        # client_nbr=i,
    )
    fl.client.start_numpy_client("[::]:8080", client=client)
