import Federated_set
import tensorflow as tf

from Preparing_Data import X_test,y_test


(X_train_1,y_train_1)  = Federated_set.Set[0][0]

print(X_train_1.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (74,)),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)


model.evaluate(X_test,
        y_test)