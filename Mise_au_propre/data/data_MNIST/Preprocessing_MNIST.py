import tensorflow as tf


def Data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = tf.reshape(X_train, X_train.shape + (1,))
    X_test = tf.reshape(X_test, X_test.shape + (1,))

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = Data()
