import os
import sys
import matplotlib.pyplot as plt

PATH_MODEL = "/home/hugo/hugo/Stage/Mise_au_propre/Model/"
PATH_DATA = "/home/hugo/hugo/Stage/Mise_au_propre/data/data_CIC_IDS2017/"

sys.path.insert(1, PATH_MODEL)
sys.path.insert(1, PATH_DATA)


from model_CIC_IDS2017 import create_model_CIC_IDS2017

# from Preprocessing_CIC_IDS2017 import X_train, X_test, y_train, y_test
