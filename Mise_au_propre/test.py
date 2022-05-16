def run_centralized_JS(
    epochs,
):
    import pickle
    from Model.model_JS import create_model_JS
    from data.data_JS.Preprocessing_JS import X_test, X_train, y_test, y_train

    model = create_model_JS()
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_test, y_test)
    )
    print(history.history)
    dict = history.history
    for keys in list(dict.keys()):
        if "val" not in keys:
            dict.pop(keys)
    print(dict)
    """
    file_name = "centralized"
    with open(file_name, "wb") as f:
        pickle.dump(history.history, f)
"""

run_centralized_JS(1)

