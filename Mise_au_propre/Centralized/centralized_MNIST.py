def run_centralized_MNIST(epochs):
    from Model.model_MNIST import create_model_MNIST
    from data.data_MNIST.Preprocessing_MNIST import X_train, X_test, y_train, y_test

    model = create_model_MNIST()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        batch_size=32,
    )
