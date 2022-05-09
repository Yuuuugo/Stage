import FLconfig


def run_centralized_IMDB(epochs):
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
