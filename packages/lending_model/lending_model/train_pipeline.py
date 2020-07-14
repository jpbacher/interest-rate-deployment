import logging

import pipeline
from config import config
from processing.data_management import (
    load_dataset, split_dataset, save_pipeline, clean_target_variable
)
from lending_model import __version__ as _version


_logger = logging.getLogger(__name__)


def run_training():
    """Train the model."""
    # read in data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    # clean the target variable
    data = clean_target_variable(data)
    # split training & validation data
    X_train, X_val, y_train, y_val = split_dataset(data)
    print('...fitting model')
    pipeline.interest_rate_pipe.fit(X_train[config.FEATURES], y_train)

    _logger.info(f"...saving model version: {_version}")
    save_pipeline(pipeline.interest_rate_pipe)


if __name__ == '__main__':
    run_training()
