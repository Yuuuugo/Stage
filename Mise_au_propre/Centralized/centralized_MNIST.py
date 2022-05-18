def run_centralized_MNIST(epochs, nbr_clients, directory_name):
    import pickle
    from Model.model_MNIST import create_model_MNIST
    from data.data_MNIST.Preprocessing_MNIST import X_train, X_test, y_train, y_test

    print(len(X_train))
    model = create_model_MNIST()

    X_train_epochs = [[] for w in range(epochs)]
    y_train_epochs = [[] for w in range(epochs)]

    for i in range(nbr_clients):
        X_train_i = X_train[
            int((i / nbr_clients) * len(X_train)) : int(
                ((i + 1) / nbr_clients) * len(X_train)
            )
        ]
        y_train_i = y_train[
            int((i / nbr_clients) * len(y_train)) : int(
                ((i + 1) / nbr_clients) * len(y_train)
            )
        ]
        # So each client have a different dataset to train on
        for actual_rnd in range(epochs):
            X_train_i_actual_rnd = X_train_i[
                int((actual_rnd / epochs) * len(X_train_i)) : int(
                    ((actual_rnd + 1) / epochs) * len(X_train_i)
                )
            ]
            y_train_i_actual_rnd = y_train_i[
                int((actual_rnd / epochs) * len(y_train_i)) : int(
                    ((actual_rnd + 1) / epochs) * len(y_train_i)
                )
            ]

            X_train_epochs[i].append(X_train_i_actual_rnd)
            y_train_epochs[i].append(y_train_i_actual_rnd)

    print(len(X_train_epochs[0][0]))
    print(len(X_train_epochs[1][0]))

    print("SIZEE = " + str(len(X_train_epochs[0])))
    for i in range(epochs):
        print(i)
        X_t = X_train_epochs[i][0]
        y_t = y_train_epochs[i][0]

        for j in range(1, len(X_train_epochs[epochs - 1])):
            X_t = X_t + X_train_epochs[i][j]
            y_t = y_t + y_train_epochs[i][j]
        print(y_t)
        history = model.fit(
            X_t,
            y_t,
            epochs=1,
            validation_data=(X_test, y_test),
            batch_size=32,
        )

    """ history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        batch_size=32,
    ) 
    list = []
    dict = history.history
    for key in dict.keys():
        if "val" in key and "loss" not in key:  # ugly way to only select the metrics
            for i in range(len(dict[key])):
                list.append((dict[key][i], dict[key][i]))

    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(list, f)"""
