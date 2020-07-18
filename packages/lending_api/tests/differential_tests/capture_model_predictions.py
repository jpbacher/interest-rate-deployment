"""
Do not run this script locally; it will disrupt the
differential test versioning logic.
"""

import pandas as pd

from lending_model.config import config
from lending_model.predict import make_prediction
from lending_model.processing.data_management import (
    load_dataset, split_dataset, clean_target_variable
)


def capture_predictions(save_file='test_data_predictions.csv'):
    """Save test data predictions to a csv."""
    data = load_dataset(file_name='lending_data.csv')
    data = clean_target_variable(data)
    _, X_validation, _, _ = split_dataset(data)

    # take a slice with no input validation issues
    multiple_test_json = X_validation[99:600]

    predictions = make_prediction(input_data=multiple_test_json)
    predictions_df = pd.DataFrame(predictions)

    # save file to lending model package, not installed package
    predictions_df.to_csv(
        f'{config.PACKAGE_ROOT.parent}/'
        f'lending_model/lending_model/datasets/{save_file}')


if __name__ == '__main__':
    capture_predictions()
