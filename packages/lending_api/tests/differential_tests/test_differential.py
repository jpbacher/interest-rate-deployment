import math
import pytest

from lending_model.config import config
from lending_model.predict import make_prediction
from lending_model.processing.data_management import (
    load_dataset, split_dataset, clean_target_variable
)


@pytest.mark.differential
def test_model_prediction_differential(
        save_file='validation_data_predictions.csv'):
    """
    This test compares the prediction result similarity of the
    current model with the previous model's results.
    """
    # given
    previous_model_df = load_dataset('validation_data_predictions.csv')
    previous_model_predictions = previous_model_df.predictions.values
    data = load_dataset('lending_data.csv')
    data = clean_target_variable(data)
    _, X_validation, _, _ = split_dataset(data)
    multiple_test_json = X_validation[99:600]
    # when
    response = make_prediction(input_data=multiple_test_json)
    current_model_predictions = response.get('predictions')
    # then
    # difference the current model vs. the old model
    assert len(previous_model_predictions) == len(
        current_model_predictions
    )
    # perform differential test
    for previous_value, current_value in zip(
        previous_model_predictions, current_model_predictions):
        # convert numpy float to Python float
        previous_value = previous_value.item()
        current_value = current_value.item()

        assert math.isclose(previous_value,
                            current_value,
                            rel_tol=config.ACCEPTABLE_MODEL_DIFFERENCE)
