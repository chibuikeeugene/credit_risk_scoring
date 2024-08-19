from typing import Any, List, Optional

from classification_model.processing.validation import CustomerDataInputSchema
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[list[int]]


class MultipleCustomerDataInputs(BaseModel):
    inputs: List[CustomerDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "seniority": 9,
                        "home": 1,
                        "time": 60,
                        "age": 30,
                        "marital": 2,
                        "records": 1,
                        "job": 3,
                        "expenses": 73,
                        "income": 129,
                        "assets": 0,
                        "debt": 0,
                        "amount": 800,
                        "price": 846,
                    }
                ]
            }
        }
