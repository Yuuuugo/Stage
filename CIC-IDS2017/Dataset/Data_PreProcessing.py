import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


df = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', engine='python')
df.columns = df.columns.str.strip()
df = df.drop(columns=['Fwd Header Length.1'])

df = df.drop(df[pd.isnull(df['Flow ID'])].index)

df.replace('Infinity', -1, inplace=True)
df[["Flow Bytes/s", "Flow Packets/s"]] = df[["Flow Bytes/s", "Flow Packets/s"]].apply(pd.to_numeric)

df.replace([np.inf, -np.inf, np.nan], -1, inplace=True)

string_features = list(df.select_dtypes(include=['object']).columns)
string_features.remove('Label')

benign_total = len(df[df['Label'] == "BENIGN"])

attack_total = len(df[df['Label'] != "BENIGN"])

df.to_csv("web_attacks_unbalanced.csv", index=False)

enlargement = 1.1
benign_included_max = attack_total / 30 * 70
benign_inc_probability = (benign_included_max / benign_total) * enlargement
print(benign_included_max, benign_inc_probability)


import random
indexes = []
benign_included_count = 0
for index, row in df.iterrows():
    if (row['Label'] != "BENIGN"):
        indexes.append(index)
    else:
        # Copying with benign_inc_probability
        if random.random() > benign_inc_probability: continue
        # Have we achieved 70% (5087 records)?
        if benign_included_count > benign_included_max: continue
        benign_included_count += 1
        indexes.append(index)
df_balanced = df.loc[indexes]

df_balanced.to_csv("web_attacks_balanced.csv", index=False)

