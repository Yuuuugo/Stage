#!/usr/bin/python3
import pickle
import os
import matplotlib.pyplot as plt
import os


dict = {"server": {}, "centralized": {}}

experience_path = "MNIST_FedAvg_clients_2_rounds5_20220524180104"
for root, dirs, files in os.walk(experience_path + "/", topdown=False):
    print(files)
    for filename in files:

        if filename == "server":
            unpickleFile = open(experience_path + "/" + filename, "rb")
            server_metrics = pickle.load(unpickleFile, encoding="latin1")
            server_duration = pickle.load(unpickleFile, encoding="latin1")
            server_metrics.pop(
                0
            )  # We skip the evaluation round that happend before training
            dict["server"]["metrics"] = server_metrics
            dict["server"]["duration"] = server_duration

        elif filename == "centralized":
            unpickleFile = open(experience_path + "/" + filename, "rb")
            centralized_metrics = pickle.load(unpickleFile, encoding="latin1")
            centralized_duration = pickle.load(unpickleFile, encoding="latin1")

            dict["centralized"]["metrics"] = centralized_metrics
            dict["centralized"]["duration"] = centralized_duration

        elif "client_" in filename:
            unpickleFile = open(experience_path + "/" + filename, "rb")
            new_dict = pickle.load(unpickleFile, encoding="latin1")
            dict[filename] = {}
            dict[filename]["metrics"] = new_dict
            dict[filename]["duration"] = dict["server"]["duration"]


print(dict)


fig, ax = plt.subplots()


for component in list(dict.keys()):
    metrics = dict[component]["metrics"]
    duration = dict[component]["duration"]
    y = []
    for element in metrics:
        y.append(element[1])
    plt.plot(duration, y, label=component)
    plt.legend()

fig.savefig(experience_path + "/metric_with_all_client.jpg")
