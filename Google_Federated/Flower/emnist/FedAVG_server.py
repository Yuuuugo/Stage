 
import flwr as fl


from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf
from DataSet import *
from Model import * 
from util import * 


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
     
    model = CNN



    # Create strategy
    strategy = fl.server.strategy.FedAvg(
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

    # Use the last 5k training examples as a validation set

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        #model.fit(x_train,y_train,epochs = 5) Not needed 
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss,{"accuracy": accuracy} #,loss ( not really needed )

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
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
 
 


