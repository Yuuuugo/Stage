from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from Data import Data
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = Data()


model = keras.Sequential([
    layers.Flatten(input_shape = (62,9)),
    layers.Dense( units = 52, activation = 'relu'),
    layers.Dense( units = 128, activation= 'relu' ),
    layers.Dense( units = 512, activation = 'relu'),
    layers.Dense( units = 7, activation= 'linear')
    ])


model.compile(
        loss='mean_squared_error' , 
        optimizer = keras.optimizers.RMSprop(learning_rate=0.04,clipvalue=1), # clip value important to solve exploding gradient problem
        metrics=[keras.metrics.RootMeanSquaredError()]
        )


model.fit(X_train, y_train, epochs = 100 ,validation_data = (X_test,y_test))





