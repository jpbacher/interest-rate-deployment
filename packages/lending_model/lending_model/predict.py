import numpy as np
import pandas as pd

from packages.lending_model.lending_model.processing.data_management import load_pipeline
from packages.lending_model.lending_model.config import config


pipeline_file_name = 'rf_regression.pkl'
_interest_rate_pipe = load_pipeline(pipeline_file_name)


def make_prediction(input_data):
    """Make prediction using the saved model pipeline."""
    data = pd.read_json(input_data)
    output = _interest_rate_pipe.predict(data[config.FEATURES])
    response = {'Predictions': output}
    return response
