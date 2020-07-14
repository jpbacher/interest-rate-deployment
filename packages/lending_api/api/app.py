from flask import Flask


def create_app():
    """Create flask app instance."""
    flask_app = Flask('lending_api')

    # import blueprints
    from api.controller import prediction_app
    flask_app.register_blueprint(prediction_app)

    return flask_app
