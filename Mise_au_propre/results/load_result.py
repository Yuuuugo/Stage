#!/usr/bin/python3
import pickle
import os
import matplotlib.pyplot as plt
import os


def create_curves(experience_path):

    dict = {"server": {}, "centralized": {}}
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

    fig, ax = plt.subplots()

    for component in list(dict.keys()):
        if "client_" in component:
            dict[component]["duration"] = dict["server"]["duration"]
        metrics = dict[component]["metrics"]
        duration = dict[component]["duration"]
        print(duration)
        print((component, len(metrics), len(duration)))
        y = []
        for element in metrics:
            y.append(element[1])
        plt.plot(duration, y, label=component)
        plt.legend()

    fig.savefig(experience_path + "/metric_with_all_client.jpg")
