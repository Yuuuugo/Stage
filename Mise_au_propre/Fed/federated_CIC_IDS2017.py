from http import client
import os
import time
from multiprocessing import Process


from Model.model_CIC_IDS2017 import create_model_CIC_IDS2017

from Fed.Client.client_CIC_IDS2017 import Client_CIC_IDS2017
import flwr as fl
from Fed.Server.server_FedAvg import FedAvg2
from Fed.Server.server_FedAdam import FedAdam2
from Fed.Server.server_FedYogi import FedYogi2
from Fed.Server.server_FedAdagrad import FedAdagrad2


def start_server(strategy, X_test, y_test, nbr_clients, nbr_rounds):

    """Start the server with a slightly adjusted FedAvg strategy."""
    model = create_model_CIC_IDS2017()
    arguments = [model, X_test, y_test, nbr_clients, nbr_rounds]
    server = eval(strategy + "2")(*arguments)


def run_CIC_IDS2017(strategy, nbr_clients, nbr_rounds, timed):
    from data.data_CIC_IDS2017.Preprocessing_CIC_IDS2017 import (
        X_test_centralized,
        y_test_centralized,
    )

    process = []
    server_process = Process(
        target=start_server,
        args=(
            strategy,
            X_test_centralized,
            y_test_centralized,
            nbr_clients,
            nbr_rounds,
        ),
    )
    server_process.start()
    process.append(server_process)
    time.sleep(2)

    for i in range(nbr_clients):
        Client_i = Process(
            target=start_client,
            args=(
                i,
                timed,
            ),
        )
        Client_i.start()
        process.append(Client_i)

    for p in process:
        p.join()


def start_client(i, timed):
    from data.data_CIC_IDS2017.Preprocessing_CIC_IDS2017 import (
        Set,
        X_test_centralized,
        y_test_centralized,
    )

    model = create_model_CIC_IDS2017()
    client = Client_CIC_IDS2017(
        model=model,
        Set=Set,
        X_test=X_test_centralized,
        y_test=y_test_centralized,
        client_nbr=i,
        timed=timed,
    )
    fl.client.start_numpy_client("[::]:8080", client=client)
