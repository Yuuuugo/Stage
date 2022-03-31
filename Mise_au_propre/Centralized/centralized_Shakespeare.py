import os
import sys
import matplotlib.pyplot as plt

PATH_MODEL = "/home/hugo/hugo/Stage/Mise_au_propre/Model/"
PATH_DATA = "/home/hugo/hugo/Stage/Mise_au_propre/data/data_Shakespeare/"

sys.path.insert(1, PATH_MODEL)
sys.path.insert(1, PATH_DATA)

from model_Shakespeare import model
from Preprocessing_Shakespeare import dataset


EPOCHS = 30
# history = model.fit(dataset, epochs=EPOCHS)
