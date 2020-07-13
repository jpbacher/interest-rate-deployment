from lending_model.processing.data_management import (
    load_dataset, split_dataset, clean_target_variable
)
from lending_model.predict import make_prediction


def test_make_single_prediction():
    """Make a single prediction on validation data."""
    # given
    data = load_dataset(file_name='lending_data.csv')
    data = clean_target_variable(data)
    _, X_validation, _, _ = split_dataset(data)
    single_validation_json = X_validation[0:1].to_json(orient='records')
    # when
    subject = make_prediction(input_data=single_validation_json)
    # then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
    assert subject.get('predictions')[0] > 0


def test_make_multiple_predictions():
    """Make multiple predictions on validation data"""
    # given
    data = load_dataset(file_name='lending_data.csv')
    data = clean_target_variable(data)
    _, X_validation, _, _ = split_dataset(data)
    multiple_validation_json = X_validation.to_json(orient='records')
    # when
    subject = make_prediction(input_data=multiple_validation_json)
    # then
    assert subject is not None
    assert len(subject.get('predictions')) == 44358
