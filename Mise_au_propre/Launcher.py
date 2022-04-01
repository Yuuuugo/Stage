import threading
import os
import sys
import argparse
import matplotlib
from subprocess import call

PATH = os.getcwd()
""" sys.path.append(PATH + "/Centralized")
import centralized_JS
import centralized_CIC_IDS2017
import centralized_MovieLens
import centralized_Shakespeare """

from Fed.federated_JS import run_JS

""" import federated_CIC_IDS2017
import federated_MovieLens """


import traceback
import signal

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


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
        choices=["JS", "CIC-IDS_2017", "MovieLens", "CIFAR10", "Shakespeare", "emnist"],
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
    centralized = "centralized_" + args.Dataset + ".run()"

    federated = "run_" + args.Dataset
    arguments = [args.strategy, args.nbr_clients, args.nbr_rounds]

    print("-------------------" * 4 + "Start of Centralized" + "-----------------" * 4)
    # eval(centralized)
    print("-------------------" * 4 + "Start of Federated" + "-----------------" * 4)
    eval(federated)(*arguments)


if __name__ == "__main__":
    main()
