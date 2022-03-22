import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random


df = pd.read_csv('/home/hugo/hugo/Stage/CIC-IDS2017/Dataset/web_attacks_balanced.csv')
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

excluded = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
df = df.drop(columns=excluded, errors='ignore')

excluded2 = ['Init_Win_bytes_backward', 'Init_Win_bytes_forward']
df = df.drop(columns=excluded2, errors='ignore')

y = df['Label'].values
X = df.drop(columns=['Label'])
#print(X.shape, y.shape)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)