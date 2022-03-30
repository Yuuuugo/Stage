# import threading
import os
import grpc

import flwr as fl
import tensorflow as tf


from multiprocessing import Process

from own_client import Client

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(74,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)


def launching(i):
    print("Start of thread " + str(i))
    # Start Flower client
    client = Client(model=model, Set=Set[i], X_test=X_test, y_test=y_test, arg=i)
    fl.client.start_numpy_client("[::]:8080", client=client)


Process_list = []
for i in range(nb_client):
    c = Process(target=launching, args=(i,))
    Process_list.append(c)
    Process_list[i].start()
    Process_list[i].join()

""" for i in range(nb_client):
    Process_list[i].join() """

""" for Thread in Thread_list:
    Thread.start() """
