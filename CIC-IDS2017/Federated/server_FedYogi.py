from re import X
import flwr as fl


from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf
from tensorflow.keras import layers
from Data import *


import matplotlib.pyplot as plt 

Epochs = []
RMSE_value = []
Loss_value = []



def del_big_value(L):
    reduced_epochs = []
    reduced_L = []
    for i in range(len(L)):
        if L[i] < 10:
            reduced_L.append(L[i])
            reduced_epochs.append(i)
    return reduced_L, reduced_epochs

def plotting_():
    #del RMSE_value[0]
    reduced_RMSE, reduced_epochs = del_big_value(RMSE_value)
    plt.plot(reduced_epochs,reduced_RMSE)
    plt.title(" Strategy = FedYogi")
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.figtext(.6, .8, "Number of client = " +  str(nb_client))
    plt.figtext(.6, .75, "Final RMSE = " +  str(round(RMSE_value[len(RMSE_value)-1],3)))
    plt.show()

    del Loss_value[0]
    reduced_Loss,reduced_epochs = del_big_value(Loss_value)
    plt.plot(reduced_epochs,reduced_Loss)
    plt.title(" Strategy = FedYogi")
    plt.figtext(.6, .8, "Number of client = " +  str(nb_client))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.figtext(.6, .8, "Number of client = " +  str(nb_client))
    plt.figtext(.6, .75, "Final Loss = " +  str(round(RMSE_value[len(RMSE_value)-1],3)))
    plt.show()

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = tf.keras.Sequential([
    layers.Flatten(input_shape = (62,9)),
    #layers.Dense( units = 52, activation = 'relu'),
    layers.BatchNormalization(),
    #layers.Dense( units = 128, activation= 'relu' ),
    layers.BatchNormalization(),
    #layers.Dense( units = 512, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense( units = 7, activation= 'linear')
    ])

    model.compile(
        loss='mean_squared_error' , 
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01,clipvalue=1), # clip value important to solve exploding gradient problem
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    # Create strategy
    strategy = fl.server.strategy.FedYogi(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=nb_client,
        min_eval_clients=2,
        min_available_clients=nb_client,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": nb_rounds}, strategy=strategy)


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    _, x_val ,_, y_val = Data()

    # Use the last 5k training examples as a validation set

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        #model.fit(x_train,y_train,epochs = 5) Not needed 
        loss, RMSE = model.evaluate(x_val, y_val)
        RMSE_value.append(RMSE)
        Loss_value.append(loss)
        return loss,{"RMSE": RMSE} #,loss ( not really needed )

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    Epochs.append(rnd)
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
        "rnd":rnd
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 5
    return {"val_steps": val_steps}





if __name__ == "__main__":
    main()
    plotting_()

