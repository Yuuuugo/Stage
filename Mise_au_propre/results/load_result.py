import pickle
import os
import matplotlib.pyplot as plt

""" filename = "JS_FedAdam_clients_5_rounds5_20220513171847/server"

unpickleFile = open(filename, "rb")
new_dict = pickle.load(unpickleFile, encoding="latin1")
print(new_dict)
 """


import os


dict = {}

experience_path = "MNIST_FedYogi_clients_5_rounds5_20220513221833"
for root, dirs, files in os.walk(experience_path + "/", topdown=False):
    print(files)
    for filename in files:
        unpickleFile = open(experience_path + "/" + filename, "rb")
        new_dict = pickle.load(unpickleFile, encoding="latin1")
        dict[filename] = new_dict

dict["server"].pop(0)


fig, ax = plt.subplots()

rounds = [i for i in range(len(dict[list(dict.keys())[0]]))]


for i in list(dict.keys()):
    list = dict[i]
    y = []
    for elements in list:
        y.append(elements[1])
    plt.plot(rounds, y, label=i)
    plt.legend()

fig.savefig(experience_path + ".jpg")
