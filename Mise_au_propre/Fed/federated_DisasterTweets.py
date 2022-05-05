#!/usr/bin/python3
import time
from multiprocessing import Process


from Fed.Client.client import Client_Test
import flwr as fl

from Fed.Server.server_FedAvg import FedAvg2
from Fed.Server.server_FedAdam import FedAdam2
from Fed.Server.server_FedYogi import FedYogi2
from Fed.Server.server_FedAdagrad import FedAdagrad2


def start_server(strategy, X_test, y_test, nbr_clients, nbr_rounds):
    from Model.model_DisasterTweets import create_model_DisasterTweets
    from data.data_DisasterTweets.Preprocessing_DisasterTweets import X_test, y_test

    """Start the server with a slightly adjusted FedAvg strategy."""
    model = create_model_DisasterTweets()
    arguments = [model, X_test, y_test, nbr_clients, nbr_rounds]
    server = eval(strategy + "2")(*arguments)


def run_DisasterTweets(strategy, nbr_clients, nbr_rounds, timed):
    from data.data_DisasterTweets.Preprocessing_DisasterTweets import (
        X_test,
        X_train,
        y_test,
        y_train,
    )

    process = []
    # model2 = deepcopy(create_model_JS()) Bug
    server_process = Process(
        target=start_server,
        args=(strategy, X_test, y_test, nbr_clients, nbr_rounds),
    )
    # server_process = Process(target=start_server, args=(nbr_rounds, nbr_clients, 0.2))
    server_process.start()
    process.append(server_process)
    time.sleep(2)

    print("After start")
    for i in range(nbr_clients):
        Client_i = Process(
            target=start_client,
            args=(
                i,
                timed,
                nbr_clients,
            ),
        )
        Client_i.start()
        process.append(Client_i)

    for p in process:
        p.join()


def start_client(i, timed, nbr_clients):
    from data.data_DisasterTweets.Preprocessing_DisasterTweets import (
        X_test,
        X_train,
        y_test,
        y_train,
    )
    from Model.model_DisasterTweets import create_model_DisasterTweets

    X_train[
        int((i / nbr_clients) * len(X_train)) : int(
            ((i + 1) / nbr_clients) * len(X_train)
        )
    ],
    y_train[
        int((i / nbr_clients) * len(y_train)) : int(
            ((i + 1) / nbr_clients) * len(y_train)
        )
    ],  # So each client have a different dataset to train on

    print("Launching of client" + str(i))
    # Start Flower client
    model = create_model_DisasterTweets()
    client = Client_Test(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        client_nbr=i,
        timed=timed,
    )
    fl.client.start_numpy_client("[::]:8080", client=client)
