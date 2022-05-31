#!/bin/bash

for i in range 5
do
	for dataset in Bostonhouse    
	do
		for strategy in FedAvg FedYogi FedAdam FedAdagrad
		do
			for nbr_clients in 3 5 7 10 20
			do
				for nbr_rounds in 5 10 20 25 50 100
				do
					python3 Launcher.py --Dataset=$dataset --strategy=$strategy --nbr_clients=$nbr_clients --nbr_rounds=$nbr_rounds
				done	
			done
		done
	done
done


