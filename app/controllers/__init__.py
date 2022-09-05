from controllers import routes


def init(app):
    app.register_blueprint(routes.ml)
