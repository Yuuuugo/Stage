def run_centralized_MovieLens(epochs, directory_name):
    import pickle
    from Model.model_MovieLens import create_model_JS
    from data.data_MovieLens.Preprocessing_MovieLens import (
        X_test,
        X_train,
        y_test,
        y_train,
    )

    model = create_model_JS()
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_test, y_test)
    )
    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(history, f)
