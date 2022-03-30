import threading
import os
import sys
import argparse
import matplotlib
from subprocess import call


sys.path.append("/home/hugo/hugo/Stage/Mise_au_propre/Centralized")
import centralized_JS
import centralized_CIC_IDS2017
import centralized_MovieLens

sys.path.append("/home/hugo/hugo/Stage/Mise_au_propre/Federated")
import federated_JS
import federated_CIC_IDS2017
import federated_MovieLens


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--nbr_clients", type=int, choices=range(1, 100), required=True)
    parser.add_argument("--nbr_rounds", type=int, choices=range(1, 100), required=True)
    parser.add_argument(
        "--Dataset",
        type=str,
        choices=["JS", "CIC-IDS_2017", "MovieLens", "CIFAR10"],
        required=True,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["FedAvg", "FedAdam", "FedAdagrad", "FedYogi"],
        required=True,
    )
    args = parser.parse_args()

    centralized = "centralized_" + args.Dataset + ".run()"

    federated = "federated_" + args.Dataset + ".run"
    arguments = [args.strategy, args.nbr_clients, args.nbr_rounds]

    print("-------------------" * 4 + "Start of Centralized" + "-----------------" * 4)
    eval(centralized)
    print("-------------------" * 4 + "Start of Federated" + "-----------------" * 4)
    eval(federated)(*arguments)


if __name__ == "__main__":
    main()
