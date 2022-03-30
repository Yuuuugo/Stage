from pickletools import optimize
import tensorflow as tf


Resnet = tf.keras.applications.ResNet50(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)

for layer in Resnet.layers:
    layer.trainable = False

inputs = tf.keras.layers.Input(shape=(32, 32, 3))
x = tf.keras.layers.UpSampling2D((7, 7))(inputs)
x = Resnet(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=1024, activation="relu")(x)
x = tf.keras.layers.Dense(units=512, activation="relu")(x)
output = tf.keras.layers.Dense(units=10, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=output)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
