from flask import Flask
import logging

from src.resource.config import Config
from presentation.controllers import routes


app = Flask(__name__)
app.config.from_object(Config)

logging.getLogger('waitress').setLevel(logging.INFO)


def get_app():
    app.register_blueprint(routes.ml)
    return app
