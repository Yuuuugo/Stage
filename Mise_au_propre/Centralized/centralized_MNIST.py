def run_centralized_MNIST(epochs, directory_name):
    import pickle
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
    list = []
    dict = history.history
    for key in dict.keys(): 
        if "val" in key and "loss" not in key: #ugly way to only select the metrics
            for i in range(len(dict[key])):
                list.append((dict[key][i], dict[key][i]))
    
    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(list, f)
    
    
