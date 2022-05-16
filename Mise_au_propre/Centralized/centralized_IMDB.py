def run_centralized_IMDB(epochs, directory_name):
    import pickle
    from Model.model_IMDB import create_model_IMDB
    from data.data_IMDB.Preprocessing_IMDB import (
        X_train,
        X_test,
        y_train,
        y_test,
    )

    model = create_model_IMDB()
    history = model.fit(
        [X_train], y_train, epochs=epochs, validation_data=(X_test, y_test)
    )
    dict = history.history
    for keys in list(dict.keys()):
        if "val" not in keys:
            dict.pop(keys)

    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(dict, f)
