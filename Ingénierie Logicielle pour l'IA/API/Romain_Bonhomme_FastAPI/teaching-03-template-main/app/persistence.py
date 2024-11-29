# This file will contain the methods to persist
# the trained model:
# save_model(parameters: model, filename)
# load_model(parameters: filename)

from joblib import dump, load
from pathlib import Path


def save_model(model, filename="iris_model.joblib"):
    """Save the trained model to the specified file."""
    save_path = Path("models/")
    save_path.mkdir(parents=True, exist_ok=True)
    dump(model, save_path / filename)


def load_model(filename="iris_model.joblib"):
    """Load the model from the specified file."""
    model_path = Path("models/") / filename
    return load(model_path)
