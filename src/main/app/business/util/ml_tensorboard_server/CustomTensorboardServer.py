import tensorboard as tb
from werkzeug import serving
from werkzeug.middleware import dispatcher

from app import app
from src.main.resource.config import Config


class CustomTensorboardServer(tb.program.TensorBoardServer):

    def __init__(self, tensorboard_app, flags):
        self._app = dispatcher.DispatcherMiddleware(app.app, {"/tensorboard": tensorboard_app})

    def serve_forever(self):
        serving.run_simple('0.0.0.0', Config.SERVER_PORT, app.app)

    def get_url(self):
        return Config.SERVER_PROTOCOL + "%s:%s" % ('0.0.0.0', Config.SERVER_PORT)

    def print_serving_message(self):
        pass  # Werkzeug's `serving.run_simple` handles this
