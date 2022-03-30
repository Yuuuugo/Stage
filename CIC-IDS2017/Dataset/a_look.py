from sqlite3 import Timestamp
import pandas as pd

PATH = "/home/hugo/hugo/Stage/CIC-IDS2017/Dataset/Unbalanced/"

df_1 = pd.read_csv(PATH + "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_unbalanced.csv")
df_2 = pd.read_csv(PATH + "Monday-WorkingHours.pcap_ISCX_unbalanced.csv")
df_3 = pd.read_csv(PATH + "Tuesday-WorkingHours.pcap_ISCX_unbalanced.csv")
df_4 = pd.read_csv(
    PATH + "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_unbalanced.csv"
)
df_5 = pd.read_csv(
    PATH + "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX_unbalanced.csv"
)
df_6 = pd.read_csv(
    PATH + "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_unbalanced.csv"
)
df_7 = pd.read_csv(PATH + "Wednesday-workingHours.pcap_ISCX_unbalanced.csv")
df_8 = pd.read_csv(PATH + "Friday-WorkingHours-Morning.pcap_ISCX_unbalanced.csv")

Set = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8]

for df in Set:
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    Web_server_16_Public = pd.concat(
        [
            df[df["Destination IP"] == "192.168.10.50"],
            df[df["Destination IP"] == "205.174.165.68"],
        ]
    )
    Ubuntu_server_12_Public = pd.concat(
        [
            df[df["Destination IP"] == "192.168.10.51"],
            df[df["Destination IP"] == "205.174.165.66"],
        ]
    )
    Ubuntu_14_4_32B = df[df["Destination IP"] == "192.168.10.19"]
    Ubuntu_14_4_64B = df[df["Destination IP"] == "192.168.10.17"]
    Ubuntu_16_4_32B = df[df["Destination IP"] == "192.168.10.16"]
    Ubuntu_16_4_64B = df[df["Destination IP"] == "192.168.10.12"]
    Win_7_Pro_64B = df[df["Destination IP"] == "192.168.10.9"]
    Win_8_1_64B = df[df["Destination IP"] == "192.168.10.5"]
    Win_Vista_64B = df[df["Destination IP"] == "192.168.10.8"]
    Win_10_pro_32B = df[df["Destination IP"] == "192.168.10.14"]
    Win_10_64B = df[df["Destination IP"] == "192.168.10.15"]
    MACe = df[df["Destination IP"] == "192.168.10.25"]
    Insiders = [
        Web_server_16_Public,
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
        MACe,
    ]
    excluded = [
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Protocol",
    ]
    excluded2 = ["Init_Win_bytes_backward", "Init_Win_bytes_forward"]
    for i in range(len(Insiders)):
        Insiders[i] = Insiders[i].drop(columns=excluded, errors="ignore")
        Insiders[i] = Insiders[i].drop(columns=excluded2, errors="ignore")
