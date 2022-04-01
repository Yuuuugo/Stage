import tensorflow as tf


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()


def preprocess_image_input(input_images):
    input_images = input_images.astype("float32")
    output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_ims


X_train = preprocess_image_input(X_train)
X_test = preprocess_image_input(X_test)
