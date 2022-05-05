import tensorflow as tf


def create_model_DisasterTweets():
    from data.data_DisasterTweets.Preprocessing_DisasterTweets import X_train

    max_vocab_length = 10000
    max_length = 15
    text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=max_vocab_length,
        output_mode="int",
        output_sequence_length=max_length,
    )

    # Fit the text vectorizer to the training text
    text_vectorizer.adapt(X_train)

    # Creating a embedding layer

    embedding = tf.keras.layers.Embedding(
        input_dim=max_vocab_length,
        output_dim=128,
        embeddings_initializer="uniform",
        input_length=max_vocab_length,
    )

    model_DisasterTweets = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,), dtype="string"),
            text_vectorizer,
            embedding,
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model_DisasterTweets.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model_DisasterTweets
