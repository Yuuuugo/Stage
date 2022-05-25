#!/bin/bash

for dataset in MNIST 
do
	for strategy in FedAvg FedYogi FedAdam FedAdagrad
	do
		for nbr_clients in 5 10 20
		do
			for nbr_rounds in 50 100
			do
				python3 Launcher.py --Dataset=MNIST --strategy=$strategy --nbr_clients=$nbr_clients --nbr_rounds=$nbr_rounds
			done

	done	done
done


'
for dataset in JS 
do
	for strategy in FedAvg FedYogi FedAdam FedAdagrad
	do
		for nbr_clients in 5 10 20 
		do
			for nbr_rounds in 5 10 20
			do
				python3 Launcher.py --Dataset=$dataset --strategy=$strategy --nbr_clients=$nbr_clients --nbr_rounds=$nbr_rounds
			done
		done
	done
done
'

'Dataset to add : 
    - CIC-IDS2017 
    - DisasterTweets (try the LSTM model)
    - '
