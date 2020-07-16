import json

from lending_model.config import config
from lending_model.processing.data_management import load_dataset, split_dataset


def test_prediction_endpoint_validation_200(flask_test_client):
    # given
    # get validation data from lending_model package
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    _, X_validation, _, _ = split_dataset(data)
    post_json = X_validation.to_json(orient='records')
    # when
    response = flask_test_client.post('/v1/predict/lending_rate',
                                      json = json.loads(post_json))
    # then
    assert response.status_code == 200
