"""def create_model_DisasterTweets():

    import tensorflow as tf
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
"""

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text


bert_model_name = "small_bert/bert_en_uncased_L-4_H-512_A-8"


tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1"
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

print(f"BERT model selected           : {tfhub_handle_encoder}")
print(f"Preprocess model auto-selected: {tfhub_handle_preprocess}")


def create_model_DisasterTweets():
    def build_classifier_model():
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        preprocessing_layer = hub.KerasLayer(
            tfhub_handle_preprocess, name="preprocessing"
        )
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(
            tfhub_handle_encoder, trainable=True, name="BERT_encoder"
        )
        outputs = encoder(encoder_inputs)
        net = outputs["pooled_output"]
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(net)
        return tf.keras.Model(text_input, net)

    classifier_model = build_classifier_model()

    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = tf.metrics.BinaryAccuracy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return classifier_model 
