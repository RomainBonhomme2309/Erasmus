# this file contains the services needed by your
# main application.
# get_or_train_model(): returns model
# predict_iris_species(parameters: input_data): returns int
# retrain_model()
# also contains the basic reporting functionnality

import numpy as np
from datetime import datetime
from model import train_model, predict_species
from persistence import load_model, save_model

# Initialize reporting dictionary
reporting = {
    "last_trained": None,
    "accuracy": None,
    "prediction_count": 0,
}


def get_or_train_model():
    """Load an existing model or train a new one if not found."""
    try:
        model = load_model()
    except FileNotFoundError:
        model, accuracy = train_model()
        save_model(model)
        # Update reporting
        reporting["last_trained"] = datetime.now().isoformat()
        reporting["accuracy"] = accuracy
    return model


def predict_iris_species(input_data):
    """Predict the Iris species based on input data."""
    model = get_or_train_model()
    features = np.array(list(input_data.dict().values())).reshape(1, -1)
    prediction = predict_species(model, features)
    # Update prediction count in reporting
    reporting["prediction_count"] += 1
    return int(prediction[0])


def retrain_model():
    """Retrain the model and save the updated version."""
    new_model, new_accuracy = train_model()
    save_model(new_model)
    # Update reporting
    reporting["last_trained"] = datetime.now().isoformat()
    reporting["accuracy"] = new_accuracy
    return new_accuracy


def get_model_report():
    """Retrieve basic reporting details about the model."""
    return reporting
