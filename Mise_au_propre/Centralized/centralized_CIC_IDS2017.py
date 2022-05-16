def run_centralized_CIC_IDS2017(epochs, directory_name):
    import pickle
    from Model.model_CIC_IDS2017 import create_model_CIC_IDS2017
    from data.data_CIC_IDS2017.experiment import (
        X_train_centralized,
        X_test_centralized,
        y_train_centralized,
        y_test_centralized,
    )

    model = create_model_CIC_IDS2017()

    history = model.fit(
        X_train_centralized,
        y_train_centralized,
        epochs=epochs,
        batch_size=32,
        verbose=1,
        validation_data=(X_test_centralized, y_test_centralized),
    )
    dict = history.history
    for keys in list(dict.keys()):
        if "val" not in keys:
            dict.pop(keys)

    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(history, f)
