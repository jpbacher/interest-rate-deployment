import pandas as pd
import logging

from lending_model.processing.data_management import load_pipeline
from lending_model.config import config
from lending_model.processing.validation import validate_inputs
from lending_model import __version__ as _version


_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_interest_rate_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(input_data):
    """Make prediction using the saved model pipeline."""
    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)
    output = _interest_rate_pipe.predict(validated_data[config.FEATURES])

    results = {"predictions": output, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs': {validated_data} "
        f"Predictions: {results}"
    )

    return results
