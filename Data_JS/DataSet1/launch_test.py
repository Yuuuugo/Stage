from multiprocessing import Process
import os
import grpc
from Data import nb_client
import tensorflow as tf
import flwr as fl
from tensorflow.keras import layers
from modified_client import Client
from Data import Data

X_train, X_test, y_train, y_test = Data()
tf.random.set_seed(42)

if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]

model = tf.keras.Sequential(
    [
        layers.Flatten(input_shape=(62, 9)),
        layers.Dense(units=52, activation="relu"),
        layers.Dense(units=128, activation="relu"),
        layers.Dense(units=512, activation="relu"),
        layers.Dense(units=7, activation="linear"),
    ]
)

model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=0.01, clipvalue=1
    ),  # clip value important to solve exploding gradient problem
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)


def launching(i):
    print("Start of thread " + str(i))
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


Process_list = []
for i in range(nb_client):
    c = Process(target=launching, args=(i,))
    Process_list.append(c)
    Process_list[i].start()
    # Process_list[i].join()

for i in range(nb_client):
    Process_list[i].join()
