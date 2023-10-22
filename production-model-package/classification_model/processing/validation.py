from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

# from classification_model.config.core import config
# from classification_model.processing.data_manager import data_preparation


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """check model inputs for na values and filter"""
    validated_data = input_data.copy()
    validated_data.dropna(inplace=True)
    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """check model inputs for unprocessable values."""
    prevalidated_data = drop_na_inputs(input_data=input_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate input
        MultipleCustomerDataInputs(
            inputs=prevalidated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return prevalidated_data, errors


class CustomerDataInputSchema(BaseModel):
    seniority: Optional[int]
    home: Optional[int]
    time: Optional[int]
    age: Optional[int]
    marital: Optional[int]
    records: Optional[int]
    job: Optional[int]
    expenses: Optional[int]
    income: Optional[int]
    assets: Optional[int]
    debt: Optional[int]
    amount: Optional[int]
    price: Optional[int]


class MultipleCustomerDataInputs(BaseModel):
    inputs: List[CustomerDataInputSchema]
