def run_centralized_CIC_IDS2017(epochs):
    from Model.model_CIC_IDS2017 import create_model_CIC_IDS2017
    from data.data_CIC_IDS2017.experiment import (
        X_train_centralized,
        X_test_centralized,
        y_train_centralized,
        y_test_centralized,
    )

    model = create_model_CIC_IDS2017()

    model.fit(
        X_train_centralized,
        y_train_centralized,
        epochs=epochs,
        batch_size=32,
        verbose=1,
        validation_data=(X_test_centralized, y_test_centralized),
    )
