# history = model.fit(dataset, epochs=EPOCHS)


def run_centralized_Shakespeare(epochs):
    from Model.model_Shakespeare import create_model_Shakespeare
    from data.data_Shakespeare.Preprocessing_Shakespeare import dataset

    model = create_model_Shakespeare()
    history = model.fit(dataset, epochs=epochs, verbose=1)
