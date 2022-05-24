from multiprocessing import Process
import time
import flwr as fl
from typing import Any, Callable, Dict, List, Optional, Tuple
from flwr.server.strategy import FedAvg
import matplotlib.pyplot as plt
import pickle
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes


def get_eval_fn(model, X_test, y_test, list_metrics, duration):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself

    # The `evaluate` function will be called after every round

    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        duration.append(time.time())
        model.set_weights(weights)  # Update model2 with the latest parameters
        print("Test evaluate")
        loss, metrics_used = model.evaluate(X_test, y_test)
        list_metrics.append((loss, metrics_used))
        print("Test after evaluate")
        return loss, {"other metrics": metrics_used}  # ,loss ( not really needed )

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {"batch_size": 32, "local_epochs": 1, "rnd": rnd}
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


class FedAvg2(fl.server.strategy.FedAvg, Process):
    def __init__(self, model, X_test, y_test, nbr_clients, nbr_rounds, directory_name):

        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.nbr_clients = nbr_clients
        self.nbr_rounds = nbr_rounds
        self.model = model
        self.directory_name = directory_name
        self.duration = []
        self.run()

    def run(self):
        list_metrics = []
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_eval=1,
            min_fit_clients=self.nbr_clients,
            min_eval_clients=self.nbr_clients,
            min_available_clients=self.nbr_clients,
            eval_fn=get_eval_fn(
                self.model,
                self.X_test,
                self.y_test,
                list_metrics,
                self.duration,
            ),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=None,
        )

        print("Before server")
        fl.server.start_server(
            "[::]:8080", config={"num_rounds": self.nbr_rounds}, strategy=strategy
        )
        print("server " + str(list_metrics))
        file_name = self.directory_name + "/server"
        with open(file_name, "wb") as f:
            pickle.dump(list_metrics, f)
            pickle.dump(self.duration, f)
