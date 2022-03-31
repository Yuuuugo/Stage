import threading
import os
import sys
import argparse
import matplotlib
from subprocess import call

PATH = os.getcwd()
sys.path.append(PATH + "/Centralized")
import centralized_JS
import centralized_CIC_IDS2017
import centralized_MovieLens
import centralized_Shakespeare

sys.path.append(PATH + "/Federated")
import federated_JS
import federated_CIC_IDS2017
import federated_MovieLens


import traceback
import signal


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
        "--nbr_clients", type=int, choices=range(1, 100), required=True, default=3
    )
    parser.add_argument(
        "--nbr_rounds", type=int, choices=range(1, 100), required=True, default=3
    )
    parser.add_argument(
        "--Dataset",
        type=str,
        choices=["JS", "CIC-IDS_2017", "MovieLens", "CIFAR10", "Shakespeare", "emnist"],
        required=True,
        default="CIFAR10",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["FedAvg", "FedAdam", "FedAdagrad", "FedYogi"],
        required=True,
        strategy="FedAvg",
    )
    args = parser.parse_args()

    centralized = "centralized_" + args.Dataset + ".run()"

    federated = "federated_" + args.Dataset + ".run_" + args.Dataset
    arguments = [args.strategy, args.nbr_clients, args.nbr_rounds]

    print("-------------------" * 4 + "Start of Centralized" + "-----------------" * 4)
    # eval(centralized)
    print("-------------------" * 4 + "Start of Federated" + "-----------------" * 4)
    eval(federated)(*arguments)


if __name__ == "__main__":
    main()
