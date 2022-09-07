from flask import Flask
import logging
import presentation.controllers as controllers


app = Flask(__name__)
app.config.from_object('config.Config')

logging.getLogger('waitress').setLevel(logging.INFO)


def get_app():
    controllers.init(app)
    return app
