import tensorflow as tf


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()


def preprocess_image_input(input_images):
    input_images = input_images.astype("float32")
    output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_ims


X_train = preprocess_image_input(X_train)
X_test = preprocess_image_input(X_test)

print(type(y_train))
""" train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

train_ds = tf.keras.preprocessing.image.NumpyArrayIterator(
    x=X_train,
    y=y_train,
    image_data_generator=train_datagen,
    batch_size=32,
)

test_ds = tf.keras.preprocessing.image.NumpyArrayIterator(
    x=X_test,
    y=y_test,
    image_data_generator=test_datagen,
    batch_size=32,
)
 """
