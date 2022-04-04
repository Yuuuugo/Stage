import tensorflow as tf


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


shape = X_train.shape

X_train = tf.reshape(X_train, X_train.shape + (1,))
X_test = tf.reshape(X_test, X_test.shape + (1,))

print(y_train)
