from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from Data import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

epochs = 10
epochs_list = [i for i in range(1,epochs+1)]

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


history = model.fit(X_train, y_train, epochs = epochs ,validation_data = (X_test,y_test))



plt.plot(epochs_list, 
        history.history["val_root_mean_squared_error"]
        )
plt.xlabel(" Epochs ")
plt.ylabel(" RMSE")
plt.title("Centralized RMSE")
plt.figtext(.6, .75, "Final RMSE = " +  str(round(history.history["val_root_mean_squared_error"][epochs-1],3)))

plt.show()




