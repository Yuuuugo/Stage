import os
from multiprocessing import Process

import flwr as fl
from typing import Any, Callable, Dict, List, Optional, Tuple
from flwr.server.strategy import FedAvg


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_eval_fn(model2, X_test, y_test):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model2 here to avoid the overhead of doing it in `evaluate` itself

    # Use the last 5k training examples as a validation set

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        model2.set_weights(weights)  # Update model2 with the latest parameters
        # model2.fit(X_test, y_test, epochs=5)
        print("Test evaluate")
        # model2.summary()
        loss, metrics_used = model2.evaluate(X_test, y_test)
        print("Test after evaluate")
        return loss, {"other metrics": metrics_used}  # ,loss ( not really needed )

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {"batch_size": 32, "local_epochs": 1 if rnd < 2 else 2, "rnd": rnd}
    return config


def evaluate_config(rnd: int):
    print("evaluate_config")
    """Return evaluation configuration dict for each round.

            Perform five local evaluation steps on each client (i.e., use five
            batches) during rounds one to three, then increase to ten local
            evaluation steps.
            """
    val_steps = 5 if rnd < 4 else 5
    return {"val_steps": val_steps}


class FedAvg2(Process):
    def __init__(self, model2, X_test, y_test, nbr_clients, nbr_rounds):
        print("Test init")
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.nbr_clients = nbr_clients
        self.nbr_rounds = nbr_rounds
        self.model = model2
        # self.client_nbr = client_nbr
        self.run()

    def run(self):

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=0.3,
            fraction_eval=0.2,
            min_fit_clients=self.nbr_clients,
            min_eval_clients=2,
            min_available_clients=self.nbr_clients,
            eval_fn=get_eval_fn(self.model, self.X_test, self.y_test),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=fl.common.weights_to_parameters(
                self.model.get_weights()
            ),
        )
        print("Before server")
        fl.server.start_server(
            "[::]:8080", config={"num_rounds": self.nbr_rounds}, strategy=strategy
        )
