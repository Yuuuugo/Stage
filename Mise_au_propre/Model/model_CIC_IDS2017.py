import tensorflow as tf


def create_model_CIC_IDS2017():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(74,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model
