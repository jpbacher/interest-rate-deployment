import json

from lending_model.config import config as model_config
from lending_model.processing.data_management import load_dataset, split_dataset
from lending_model import __version__ as _version

from api import __version__ as api_version


def test_health_endpoint_returns_200(flask_test_client):
    # when
    response = flask_test_client.get('/health')
    # then
    assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):
    # when
    response = flask_test_client.get('/version')
    # then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == _version
    assert response_json['api_version'] == api_version


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # given
    # load data from lending model package, & get validation set
    data = load_dataset(file_name=model_config.TRAINING_DATA_FILE)
    _, X_validation, _, _ = split_dataset(data)
    post_json = X_validation[0:1].to_json(orient='records')

    # when
    response = flask_test_client.post('/v1/predict/lending_rate',
                                      json=post_json)
    # then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert prediction > 0
    assert response_version == _version
