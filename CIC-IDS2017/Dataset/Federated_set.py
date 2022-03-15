
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv('/Users/hugo/Stage/Stage/CIC-IDS2017/Dataset/web_attacks_balanced.csv')
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

y = pd.DataFrame(df['Label'].values, columns= ["Label"])
X_t = df.drop(columns=['Label'])
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.05, random_state=42)



df = X_train
df["Label"] = y


#print(y_test)

#print("Taille de df = " + str(len(df)))

#print(len(df["Destination IP"].value_counts()))
# Pourquoi il y a 1013 addresse IP destinataire ? Les PCs consultents des sites ?



Web_server_16_Public = pd.concat ([df[df["Destination IP"] == "192.168.10.50"],df[df["Destination IP"] == "205.174.165.68"] ] )
Ubuntu_server_12_Public =pd.concat ([df[df["Destination IP"] == "192.168.10.51"],df[df["Destination IP"] == "205.174.165.66"] ] )
Ubuntu_14_4_32B =  df[df["Destination IP"] == "192.168.10.19"]
Ubuntu_14_4_64B = df[df["Destination IP"] == "192.168.10.17"]
Ubuntu_16_4_32B = df[df["Destination IP"] == "192.168.10.16"]
Ubuntu_16_4_64B = df[df["Destination IP"] == "192.168.10.12"]
Win_7_Pro_64B = df[df["Destination IP"] == "192.168.10.9"]
Win_8_1_64B = df[df["Destination IP"] == "192.168.10.5"]
Win_Vista_64B = df[df["Destination IP"] == "192.168.10.8"]
Win_10_pro_32B = df[df["Destination IP"] == "192.168.10.14"]
Win_10_64B = df[df["Destination IP"] == "192.168.10.15"]
MACe =  df[df["Destination IP"] == "192.168.10.25"]



Insiders = [Web_server_16_Public,
    Ubuntu_server_12_Public,
    Ubuntu_14_4_32B,
    Ubuntu_14_4_64B,
    Ubuntu_16_4_32B,
    Ubuntu_16_4_64B,
    Win_7_Pro_64B,
    Win_8_1_64B,
    Win_Vista_64B,
    Win_10_pro_32B,
    Win_10_64B,
    MACe]

excluded = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']
excluded2 = ['Init_Win_bytes_backward', 'Init_Win_bytes_forward']

X_test = X_test.drop(columns=excluded,errors= 'ignore')
X_test = X_test.drop(columns=excluded2,errors= 'ignore')
X_test = X_test.drop(columns = ["Timestamp"], errors = "ignore")

X_test=np.asarray(X_test).astype(np.int)

y_test=np.asarray(y_test).astype(np.int)


for i in range(len(Insiders)):
    Insiders[i] = Insiders[i].drop(columns=excluded,errors= 'ignore')
    Insiders[i] = Insiders[i].drop(columns=excluded2,errors= 'ignore')

    Insiders[i]['Timestamp'] = Insiders[i]['Timestamp'].apply(lambda x: x[9]+x[10] if x[10] != ":" else x[9])
    


Data_PC_hour = []

#print(Insiders[0][Insiders[0]["Timestamp"] == '9' ] )

for i in range(len(Insiders)):
    Hour_separation = []
    for j in Insiders[i]['Timestamp'].unique():
        Hour_separation.append(Insiders[i][Insiders[i]['Timestamp'] == j ])
    Data_PC_hour.append(Hour_separation)

    #print(len(Insiders[i]))


#print(Data_PC_hour[0][0].columns)1

Set = []
for i in range(len(Data_PC_hour)):
    Set_i = []
    for j in range(len(Data_PC_hour[i])):
        Data_PC_hour[i][j] = Data_PC_hour[i][j].drop(columns = "Timestamp", errors = 'ignore')
        y = Data_PC_hour[i][j]['Label'].values
        X_t = Data_PC_hour[i][j].drop(columns=['Label'])
        Set_i.append([X_t,y])
    Set.append(Set_i)


""" for i in range(len(Set)):
    print("PC number "+ str(i) + " was working " + str(len(Set[i])) + "hours")
 """


#print(len(Set[0][1]))

