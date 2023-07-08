from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config

def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var not in config.model_config.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """check model inputs for unprocessable values."""

    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate input
        MultipleCustomerDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

class CustomerDataInputSchema(BaseModel):
    Status: Optional[int]
    Seniority: Optional[int]
    Home: Optional[int]     
    Time: Optional[int]     
    Age: Optional[int]     
    Marital: Optional[int]  
    Records: Optional[int] 
    Job: Optional[int]      
    Expenses: Optional[int] 
    Income: Optional[int]  
    Assets: Optional[int]  
    Debt: Optional[int]    
    Amount: Optional[int]  
    Price: Optional[int]   

class MultipleCustomerDataInputs(BaseModel):
    inputs: List[CustomerDataInputSchema]
