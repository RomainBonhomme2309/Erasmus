# This file is the main file.
# It will contain your FastAPI code
# - initialization
# - endpoints for
#   - homepage (root)
#   - hello (heartbeat)
#   - predict(parameters: input_features)
#   - train
# - code to start server
# - NOTE - you will need to create data types for input and output of critical endpoints

from fastapi import FastAPI
import uvicorn
from datetime import datetime
from schemas import InputIrisData, OutputIrisData
from services import predict_iris_species, retrain_model

# Initialize FastAPI app
app = FastAPI()


@app.get("/")
async def read_homepage():
    """Homepage that welcomes the user."""
    return {"message": "Welcome to the Iris Prediction API!"}


@app.get("/hello")
async def get_heartbeat():
    """Heartbeat endpoint returning server timestamp."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    return {"message": current_time}


@app.post("/predict", response_model=OutputIrisData)
async def predict(data: InputIrisData):
    """Predict the species based on input features."""
    prediction = predict_iris_species(data)
    return {"species": prediction}


@app.post("/train")
async def retrain():
    """Retrain the model and return the new accuracy."""
    updated_accuracy = retrain_model()
    return {"accuracy": updated_accuracy}


@app.get("/report")
async def report():
    """Endpoint to retrieve model reporting details."""
    from services import get_model_report
    return get_model_report()


if __name__ == "__main__":
    # Launch the FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
