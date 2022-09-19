from paste.translogger import TransLogger
from waitress import serve

from src.main import app
from src.main.resource.config import Config

if __name__ == '__main__':
    serve(TransLogger(app.get_app(), setup_console_handler=True), host='0.0.0.0', port=Config.SERVER_PORT)
