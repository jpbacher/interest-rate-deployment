from config import config
from processing.data_management import load_dataset, split_dataset, save_pipeline
import pipeline


def run_training():
    """Train the model."""
    # read in data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    # split training & validation data
    X_train, X_val, y_train, y_val = split_dataset(data)
    print('...fitting model')
    pipeline.interest_rate_pipe.fit(X_train[config.FEATURES], y_train)
    print('...model fitted & saved')
    save_pipeline(pipeline.interest_rate_pipe)


if __name__ == '__main__':
    run_training()
