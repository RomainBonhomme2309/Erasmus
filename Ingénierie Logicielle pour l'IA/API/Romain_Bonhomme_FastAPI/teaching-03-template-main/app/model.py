# This file will contain code related to your model:
# training - or loading of a pre-trained model
# prediction(parameters: model, data)
# maybe reporting, if you have time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def train_model():
    """Train an XGBoost classifier on the Iris dataset."""
    iris_data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris_data.data, iris_data.target, test_size=0.2, stratify=iris_data.target, random_state=42
    )
    
    model = XGBClassifier(
        n_estimators=100, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, test_accuracy


def predict_species(model, features):
    """Predict the species using the trained model and input features."""
    return model.predict(features)
