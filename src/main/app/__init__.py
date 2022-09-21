import logging

from flask import Flask
from flask_wtf.csrf import CSRFProtect

from presentation.controllers import mnist, audio
from src.main.resource.config import Config

app = Flask(__name__)

logging.getLogger('waitress').setLevel(logging.INFO)


def get_app():
    app.config.from_object(Config)
    csrf = CSRFProtect()
    csrf.init_app(app)
    app.register_blueprint(csrf.exempt(mnist.mnist_blueprint))
    app.register_blueprint(csrf.exempt(audio.audio_blueprint))
    return app
