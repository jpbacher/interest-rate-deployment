from ..lending_model.processing.data_management import (
    load_dataset, split_dataset
)
from ..lending_model.predict import make_prediction


def test_make_single_prediction():
    """Make a single prediction on validation data."""
    # given
    data = load_dataset(file_name='lending_data.csv')
    _, X_validation, _, _ = split_dataset(data)
    single_validation_input = X_validation[0:1]
    # when
    subject = make_prediction(input_data=single_validation_input)
    # then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
    assert subject > 0
