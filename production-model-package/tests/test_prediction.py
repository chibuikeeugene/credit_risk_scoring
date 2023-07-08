import math

import numpy as np

from classification_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = 1
    expected_no_of_prediction = 1336

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.int)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_of_prediction