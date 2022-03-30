import sys
from tabnanny import verbose
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
""" 'The 2022-03-21 11:15:59.315959: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: 
CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected' error is normal """

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/hugo/hugo/Stage/CIC-IDS2017/Dataset')

from Data import X_train, X_test, y_train, y_test 
from Model import model
import tensorflow as tf

print(X_train.shape, y_train.shape)


excluded = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']
excluded2 = ['Init_Win_bytes_backward', 'Init_Win_bytes_forward']

X_train = X_train.drop(columns=excluded,errors= 'ignore')
X_train = X_train.drop(columns=excluded2,errors= 'ignore')
X_train = X_train.drop(columns = ["Timestamp"], errors = "ignore")


X_train=np.asarray(X_train).astype(np.int32)

y_train=np.asarray(y_train).astype(np.int32)


model.fit(  x = X_train,
            y = y_train,
            #validation_data = (X_test,y_test),
            epochs = 3,
            verbose = 1,
            batch_size = 32
            )
 
                  

model.evaluate(X_train, y_train)