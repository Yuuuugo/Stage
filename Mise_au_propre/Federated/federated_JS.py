#!/usr/bin/python3
import os
import sys
from xmlrpc import client
from grpc import server

import matplotlib.pyplot as plt
from multiprocessing import Process

PATH_MODEL = "/home/hugo/hugo/Stage/Mise_au_propre/Model/"
PATH_DATA = "/home/hugo/hugo/Stage/Mise_au_propre/data/data_JS/"
PATH_STRATEGY = "/home/hugo/hugo/Stage/Mise_au_propre/Federated/Server"

sys.path.insert(1, PATH_MODEL)
sys.path.insert(1, PATH_DATA)
sys.path.insert(1, PATH_STRATEGY)

from model_JS import model
from Preprocessing_JS import X_test, X_train, y_test, y_train
from Client.client import Client
import flwr as fl

import FedAvg
import FedAdagrad
import FedAdam
import FedYogi


if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


def run(strategy, nbr_clients, nbr_rounds):

    arguments_server = [model, X_test, y_test, nbr_clients, nbr_rounds]
    server_thread = Process(eval(strategy + ".run")(*arguments_server))
    server_thread.start()

    Client_list = []
    for i in range(nbr_clients):
        print(i)
        Client_i = Process(target=launching, args=(i,))
        Client_list.append(Client_i)
        Client_list[i].start()

    server_thread.join()
    for i in range(nbr_clients):
        Client_list[i].join()


def launching(i):
    print("Launching of client" + str(i))
    # Start Flower client
    client = Client(
        model=model,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        arg=i,
    )
    fl.client.start_numpy_client("[::]:8080", client=client)
