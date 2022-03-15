import sys
from tabnanny import verbose
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/hugo/Stage/Stage/CIC-IDS2017/Dataset')

from Preparing_Data import X_train, X_test, y_train, y_test 
from Model import model
import tensorflow as tf

print(X_train.shape, y_train.shape)

model.fit(  x = X_train,
            y = y_train,
            validation_data = (X_test,y_test),
            epochs = 100,
            verbose = 1,
            callbacks = [
                  tf.keras.callbacks.TensorBoard(
                      log_dir = "logs"
                  ),
            
                  ])