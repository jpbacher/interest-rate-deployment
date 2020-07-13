import pandas as pd

from lending_model.processing.data_management import load_pipeline
from lending_model.config import config


pipeline_file_name = 'rf_model_lending.pkl'
_interest_rate_pipe = load_pipeline(pipeline_file_name)


def make_prediction(input_data):
    """Make prediction using the saved model pipeline."""
    data = pd.read_json(input_data)
    output = _interest_rate_pipe.predict(data[config.FEATURES])
    response = {'predictions': output}
    return response
