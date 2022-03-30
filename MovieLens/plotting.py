import sys
import matplotlib.pyplot as plt


sys.path.insert(1, "/home/hugo/hugo/Stage/MovieLens/Federated")
from server import accuracy_value, Loss_value
from utils import nb_client


def del_big_value(L):
    reduced_epochs = []
    reduced_L = []
    for i in range(len(L)):
        if L[i] < 10:
            reduced_L.append(L[i])
            reduced_epochs.append(i)
    return reduced_L, reduced_epochs


def plotting_():
    # del accuracy_value[0]
    reduced_accuracy, reduced_epochs = del_big_value(accuracy_value)
    plt.plot(reduced_epochs, reduced_accuracy)
    plt.title(" Strategy = FedAvg")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.figtext(0.6, 0.8, "Number of client = " + str(nb_client))
    plt.figtext(
        0.6,
        0.75,
        "Final accuracy = " + str(round(accuracy_value[len(accuracy_value) - 1], 3)),
    )
    plt.show()

    del Loss_value[0]
    reduced_Loss, reduced_epochs = del_big_value(Loss_value)
    plt.plot(reduced_epochs, reduced_Loss)
    plt.title(" Strategy = FedAvg")
    plt.figtext(0.6, 0.8, "Number of client = " + str(nb_client))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.figtext(0.6, 0.8, "Number of client = " + str(nb_client))
    plt.figtext(
        0.6,
        0.75,
        "Final Loss = " + str(round(accuracy_value[len(accuracy_value) - 1], 3)),
    )
    plt.show()


plotting_()
