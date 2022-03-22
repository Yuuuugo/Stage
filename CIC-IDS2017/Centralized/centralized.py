import sys
from tabnanny import verbose

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
""" 'The 2022-03-21 11:15:59.315959: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: 
CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected' error is normal """

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/hugo/hugo/Stage/CIC-IDS2017/Dataset')

from Preparing_Data import X_train, X_test, y_train, y_test 
from Model import model
import tensorflow as tf

print(X_train.shape, y_train.shape)


model.evaluate(X_train, y_train)

model.fit(  x = X_train,
            y = y_train,
            #validation_data = (X_test,y_test),
            epochs = 20,
            verbose = 1,
            )
 
                  

model.evaluate(X_train, y_train)