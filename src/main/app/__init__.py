import logging
import os

import tensorboard as tb
from flask import Flask
from flask_wtf.csrf import CSRFProtect

from business.util.ml_tensorboard_server.CustomTensorboardServer import CustomTensorboardServer
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

    program = tb.program.TensorBoard(server_class=CustomTensorboardServer)
    program.configure(logdir=os.path.expanduser("~/data/logs"))
    # Here we should launch tb server, not in parallel thread, not as flask_app, not directly in main thread

    return app
