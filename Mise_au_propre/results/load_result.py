#!/usr/bin/python3
import pickle
import os
import matplotlib.pyplot as plt
import os


def create_curves(experience_path):

    dictonnary = {"server": {}, "centralized": {}}
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
                dictonnary["server"]["metrics"] = server_metrics
                dictonnary["server"]["duration"] = server_duration

            elif filename == "centralized":
                unpickleFile = open(experience_path + "/" + filename, "rb")
                centralized_metrics = pickle.load(unpickleFile, encoding="latin1")
                centralized_duration = pickle.load(unpickleFile, encoding="latin1")

                dictonnary["centralized"]["metrics"] = centralized_metrics
                dictonnary["centralized"]["duration"] = centralized_duration

            elif "client_" in filename:
                unpickleFile = open(experience_path + "/" + filename, "rb")
                new_dictonnary = pickle.load(unpickleFile, encoding="latin1")
                dictonnary[filename] = {}
                dictonnary[filename]["metrics"] = new_dictonnary

    fig, ax = plt.subplots()

    for component in list(dictonnary.keys()):
        if "client_" in component:
            dictonnary[component]["duration"] = dictonnary["server"]["duration"]
        metrics = dictonnary[component]["metrics"]
        duration = dictonnary[component]["duration"]
        #print(duration)
        #print((component, len(metrics), len(duration)))
        y = []
        for element in metrics:
            y.append(element[1])
        plt.plot(duration, y, label=component)

        if "JS" in experience_path:
            ax.set_ylim([0,5])
            #plt.yscale("log")
        # plt.legend()

    fig.savefig(experience_path + "/metric_with_all_client.jpg")

