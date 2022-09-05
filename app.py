from paste.translogger import TransLogger
from waitress import serve

import app

app = app.get_app()

if __name__ == '__main__':
    serve(TransLogger(app, setup_console_handler=True), host='0.0.0.0', port=5000)
