from flask import Flask
import presentation
import logging

app = Flask(__name__)
app.config.from_object('config.Config')

logging.getLogger('waitress').setLevel(logging.INFO)


def get_app():
    presentation.init(app)
    return app
