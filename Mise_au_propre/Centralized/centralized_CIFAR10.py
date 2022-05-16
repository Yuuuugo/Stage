def run_centralized_CIFAR10(epochs, directory_name):
    import pickle
    from Model.model_CIFAR10 import create_model_CIFAR10
    from data.data_CIFAR10.Preprocessing_CIFAR10 import X_train, X_test, y_test, y_train

    model = create_model_CIFAR10()
    history = model.fit(
        X_train,
        [y_train],
        epochs=epochs,
        validation_data=(X_test, [y_test]),
        batch_size=32,
    )

    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(history, f)
