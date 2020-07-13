import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from lending_model.config import config
from lending_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def load_dataset(file_name):
    """Load a dataframe."""
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return _data


def clean_target_variable(data):
    """Drop missing target data points, remove percentage signs, &
    convert to numeric."""
    data = data.dropna(axis=0, subset=[config.TARGET])
    data[config.TARGET] = data[config.TARGET].str.replace(r'%', '').astype('float')
    return data


def split_dataset(data):
    X_train, X_val, y_train, y_val = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=12
    )
    return X_train, X_val, y_train, y_val


def save_pipeline(pipeline_to_persist):
    """Persist the pipeline."""
    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = config.TRAINED_MODEL_DIR/save_file_name
    remove_old_pipelines(files_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f'saved pipeline: {save_file_name}')


def load_pipeline(file_name):
    """Load a persisted pipeline."""
    file_path = config.TRAINED_MODEL_DIR/file_name
    trained_pipeline = joblib.load(file_path)
    return trained_pipeline


def remove_old_pipelines(files_to_keep):
    """Remove old pipelines to ensure a one-to-one mapping between
    the package version & model version (to be used/imported by other
    applications.
    """
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [files_to_keep, '__init__.py']:
            model_file.unlink()
