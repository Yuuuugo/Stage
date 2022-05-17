def run_centralized_JS(epochs, directory_name):
    import pickle
    from Model.model_JS import create_model_JS
    from data.data_JS.Preprocessing_JS import X_test, X_train, y_test, y_train

    model = create_model_JS()
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_test, y_test)
    )
    list = []
    dict = history.history
    for key in dict.keys(): 
        if "val" in key and "loss" not in key: #ugly way to only select the metrics
            for i in range(len(dict[key])):
                list.append((dict[key][i], dict[key][i])) #here we need twice for the format to be in the same way as load_results need -> when I have time change it to have the loss and the metrics :
                """
        value list = []
        if "val" in key:
            # Here we need to do pairs from both of the dict keys
            list = []
            for i in dict[key]:
                list.append(i)
            value_list.append(list)
    final_list =[]
    for i in range(2*len(value_list[0])):
        if j%2 == 0:
            final_list.append(value_list[0][j/2])
        else:
            final_list.append(value_list[1][(j-1)/2])

                """

    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(list, f)
