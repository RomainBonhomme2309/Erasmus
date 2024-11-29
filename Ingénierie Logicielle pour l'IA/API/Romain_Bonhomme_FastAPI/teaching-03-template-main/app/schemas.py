# This file contains custom data types
# for input and output of critical endpoints.
# they derive from pydantic BaseModel

# class InputIrisData(BaseModel):
# class OutputIrisData(BaseModel):

from pydantic import BaseModel


class InputIrisData(BaseModel):
    """Schema for input features of the Iris dataset."""
    sepal_length: float = 5.8
    sepal_width: float = 3.0
    petal_length: float = 4.35
    petal_width: float = 1.3


class OutputIrisData(BaseModel):
    """Schema for the predicted species."""
    predicted_species: int
