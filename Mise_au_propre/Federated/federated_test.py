import os
import sys
import time
from multiprocessing import Process
from typing import Tuple

import flwr as fl
import numpy as np
import tensorflow as tf
from Client.client import Client_Test
from flwr.server.strategy import FedAvg
from flwr.server.strategy import FedYogi


PATH_MODEL = "/home/hugo/hugo/Stage/Mise_au_propre/Model/"
PATH_DATA = "/home/hugo/hugo/Stage/Mise_au_propre/data/data_JS/"
PATH_STRATEGY = "/home/hugo/hugo/Stage/Mise_au_propre/Federated/Server"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


sys.path.insert(1, PATH_MODEL)
sys.path.insert(1, PATH_DATA)
sys.path.insert(1, PATH_STRATEGY)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


from Preprocessing_JS import X_test, X_train, y_test, y_train

from model_JS import create_model_JS


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


def start_client(i) -> None:

    model = create_model_JS()

    # Start Flower client
    client = Client_Test(
        model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    fl.client.start_numpy_client("[::]:8080", client=client)


def run_simulation(num_rounds: int, num_clients: int, fraction_fit: float):
    processes = []

    # Start the server
    server_process = Process(
        target=start_server, args=(num_rounds, num_clients, fraction_fit)
    )
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(2)
    # Start all the clients
    for i in range(num_clients):
        client_process = Process(target=start_client, args=(i,))
        client_process.start()
        processes.append(client_process)

    for p in processes:
        p.join()


if __name__ == "__main__":
    run_simulation(num_rounds=10, num_clients=500, fraction_fit=0.5)
