import threading
import os
import sys
import argparse

PATH = os.getcwd()
""" sys.path.append(PATH + "/Centralized")
import centralized_JS
import centralized_CIC_IDS2017
import centralized_MovieLens

import centralized_Shakespeare """
from Centralized.centralized_JS import run_centralized_JS
from Centralized.centralized_CIFAR10 import run_centralized_CIFAR10
from Centralized.centralized_MNIST import run_centralized_MNIST
from Centralized.centralized_CIC_IDS2017 import run_centralized_CIC_IDS2017

# from Centralized.centralized_Shakespeare import


from Fed.federated_JS import run_JS
from Fed.federated_CIFAR10 import run_CIFAR10
from Fed.federated_MNIST import run_MNIST
from Fed.federated_Shakespeare import run_Shakespeare
from Fed.federated_CIC_IDS2017 import run_CIC_IDS2017

import traceback
import signal

import FLconfig

import time

actual_time = time.ctime()
actual_time = actual_time.split(" ")
timed = ""
for i in actual_time:
    timed += i + "_"


def __signal_code_to_name(code):
    for s in signal.Signals:
        if s.value == code:
            return s.name
    return "Unknown signal"


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!", __signal_code_to_name(sig))
    print(traceback.format_exc())
    # os.kill(os.getpid(), signal.SIGINT)
    sys.exit(0)


def main() -> None:
    if threading.current_thread() == threading.main_thread():
        for sig in signal.Signals:
            try:
                if sig.name == "SIGCHLD":
                    continue
                signal.signal(sig, signal_handler)
            except OSError:
                print(("Skipping signal", sig))

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")

    parser.add_argument(
        "--nbr_clients",
        type=int,
        choices=range(1, 100),
        required=True,
    )
    parser.add_argument(
        "--nbr_rounds",
        type=int,
        choices=range(1, 100),
        required=True,
    )
    parser.add_argument(
        "--Dataset",
        type=str,
        choices=["JS", "CIC_IDS2017", "MovieLens", "CIFAR10", "Shakespeare", "MNIST"],
        required=True,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["FedAvg", "FedAdam", "FedAdagrad", "FedYogi"],
        required=True,
    )
    args = parser.parse_args()
    # print(args.Dataset)
    centralized = "run_centralized_" + args.Dataset

    federated = "run_" + args.Dataset
    arguments = [args.strategy, args.nbr_clients, args.nbr_rounds, timed]

    print("-------------------" * 4 + "Start of Centralized" + "-----------------" * 4)
    # eval(centralized)(args.nbr_rounds)
    print("-------------------" * 4 + "Start of Federated" + "-----------------" * 4)
    eval(federated)(*arguments)


if __name__ == "__main__":
    main()
