#!/bin/bash

'
#JS
python3 Launcher.py --Dataset=JS --strategy=FedAvg --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=JS --strategy=FedAvg --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=JS --strategy=FedAvg --nbr_clients=20 --nbr_rounds=20

python3 Launcher.py --Dataset=JS --strategy=FedAdam --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=JS --strategy=FedAdam --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=JS --strategy=FedAdam --nbr_clients=20 --nbr_rounds=20

python3 Launcher.py --Dataset=JS --strategy=FedAdagrad --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=JS --strategy=FedAdagrad --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=JS --strategy=FedAdagrad --nbr_clients=20 --nbr_rounds=20

python3 Launcher.py --Dataset=JS --strategy=FedYogi --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=JS --strategy=FedYogi --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=JS --strategy=FedYogi --nbr_clients=20 --nbr_rounds=20
'

#MNIST
python3 Launcher.py --Dataset=MNIST --strategy=FedAvg --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=MNIST --strategy=FedAvg --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=MNIST --strategy=FedAvg --nbr_clients=20 --nbr_rounds=20

python3 Launcher.py --Dataset=MNIST --strategy=FedAdam --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=MNIST --strategy=FedAdam --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=MNIST --strategy=FedAdam --nbr_clients=20 --nbr_rounds=20

python3 Launcher.py --Dataset=MNIST --strategy=FedAdagrad --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=MNIST --strategy=FedAdagrad --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=MNIST --strategy=FedAdagrad --nbr_clients=20 --nbr_rounds=20

python3 Launcher.py --Dataset=MNIST --strategy=FedYogi --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=MNIST --strategy=FedYogi --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=MNIST --strategy=FedYogi --nbr_clients=20 --nbr_rounds=20

'
#CIFAR
python3 Launcher.py --Dataset=CIFAR10 --strategy=FedAvg --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=CIFAR10 --strategy=FedAvg --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=CIFAR10 --strategy=FedAvg --nbr_clients=20 --nbr_rounds=20

python3 Launcher.py --Dataset=CIFAR10 --strategy=FedAdam --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=CIFAR10 --strategy=FedAdam --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=CIFAR10 --strategy=FedAdam --nbr_clients=20 --nbr_rounds=20

python3 Launcher.py --Dataset=CIFAR10 --strategy=FedAdagrad --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=CIFAR10 --strategy=FedAdagrad --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=CIFAR10 --strategy=FedAdagrad --nbr_clients=20 --nbr_rounds=20

python3 Launcher.py --Dataset=CIFAR10 --strategy=FedYogi --nbr_clients=5 --nbr_rounds=5
python3 Launcher.py --Dataset=CIFAR10 --strategy=FedYogi --nbr_clients=10 --nbr_rounds=10
python3 Launcher.py --Dataset=CIFAR10 --strategy=FedYogi --nbr_clients=20 --nbr_rounds=20
'

