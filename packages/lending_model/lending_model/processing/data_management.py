import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from lending_model.config import config


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
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )
    return X_train, X_val, y_train, y_val


def save_pipeline(pipeline_to_persist):
    """Persist the pipeline."""
    save_file_name = 'rf_model_lending.pkl'
    save_path = config.TRAINED_MODEL_DIR/save_file_name
    joblib.dump(pipeline_to_persist, save_path)
    print('...saved pipeline')


def load_pipeline(file_name):
    """Load a persisted pipeline."""
    file_path = config.TRAINED_MODEL_DIR/file_name
    saved_pipeline = joblib.load(file_path)
    return saved_pipeline
