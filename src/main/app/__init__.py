from flask import Flask
import logging

from src.main.resource.config import Config
from presentation.controllers import mnist, audio


app = Flask(__name__)
app.config.from_object(Config)

logging.getLogger('waitress').setLevel(logging.INFO)


def get_app():
    app.register_blueprint(mnist.mnist_blueprint)
    app.register_blueprint(audio.audio_blueprint)
    return app
