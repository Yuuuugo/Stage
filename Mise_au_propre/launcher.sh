#!/bin/bash


python3 Launcher.py --Dataset=JS --strategy=FedAvg --nbr_clients=2 --nbr_rounds=2
#python3 Launcher.py --Dataset=MNIST --strategy=FedAvg --nbr_clients=5 --nbr_rounds=2
#python3 Launcher.py --Dataset=MNIST --strategy=FedAvg --nbr_clients=10 --nbr_rounds=2