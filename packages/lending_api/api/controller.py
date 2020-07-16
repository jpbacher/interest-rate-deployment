from flask import Blueprint, request, jsonify
from lending_model.predict import make_prediction
from lending_model import __version__ as _version

from api.config import get_logger
from api.validation import validate_inputs
from api import __version__ as api_version

_logger = get_logger(logger_name=__name__)


prediction_app = Blueprint('prediction_app', __name__)


# simple 'health' endpoint
@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})


@prediction_app.route('/v1/predict/lending_rate', methods=['POST'])
def predict():
    if request.method == 'POST':
        # extract 'post' data from request body
        json_data = request.get_json()
        _logger.info(f'Inputs: {json_data}')

        # validate input (using marshmallow schema)
        input_data, errors = validate_inputs(input_data=json_data)

        # model prediction
        result = make_prediction(input_data=input_data)
        _logger.info(f'Outputs: {result}')

        # convert array to a list
        predictions = result.get('predictions').tolist()
        version = result.get('version')

        # get the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})
