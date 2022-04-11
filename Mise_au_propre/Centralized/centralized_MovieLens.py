def run_centralized_MovieLens(epochs):
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
