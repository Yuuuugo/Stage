def run_centralized_DisasterTweets(epochs, directory_name):
    import pickle
    from Model.model_DisasterTweets import create_model_DisasterTweets
    from data.data_DisasterTweets.Preprocessing_DisasterTweets import (
        X_train,
        X_test,
        y_train,
        y_test,
    )

    model = create_model_DisasterTweets()
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_test, y_test)
    )
    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(history, f)
