import threading
import os
import sys
import grpc

sys.path.insert(1, "/home/hugo/hugo/Stage/MovieLens")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import flwr as fl
import tensorflow as tf
from utils import *
from DataSet import *
from Model import *
from own_client import Client

from multiprocessing import Process

from own_client import Client

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]

semaphore = threading.Semaphore(nb_client)


def launching(i):
    with semaphore:
        print("Start of thread " + str(i))
        # Start Flower client
        client = Client(
            model=model,
            x_train=x_train,
            y_train=y_train,
            X_test=x_val,
            y_test=y_val,
            arg=i,
        )
        fl.client.start_numpy_client("[::]:6060", client=client)


Process_list = []
for i in range(nb_client):
    c = Process(target=launching, args=(i,))
    Process_list.append(c)
    Process_list[i].start()
    # Process_list[i].join()

""" for i in range(nb_client):
    Process_list[i].join() """

""" for Thread in Thread_list:
    Thread.start() """
