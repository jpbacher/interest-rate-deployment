from .processing.data_management import load_dataset, split_dataset
from .config import config
from .pipeline import interest_rate_pipe


def run_training():
    """Train the model."""
    # read in data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    # split training & validation data
    X_train, X_val, y_train, y_val = split_dataset(data)
    interest_rate_pipe.fit(X_train[config.FEATURES], y_train)


if __name__ == '__main__':
    run_training()
