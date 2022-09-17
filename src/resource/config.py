# Define the application directory
import os


class Config(object):
    # Определяет, включен ли режим отладки
    # В случае если включен, flask будет показывать
    # подробную отладочную информацию. Если выключен -
    # - 500 ошибку без какой либо дополнительной информации.
    DEBUG = False
    MODEL_PATH = "data/models/"
    # Включение защиты против "Cross-site Request Forgery (CSRF)"
    CSRF_ENABLED = True
    # Случайный ключ, которые будет исползоваться для подписи
    # данных, например cookies.
    SECRET_KEY = 'YOUR_RANDOM_SECRET_KEY'

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # Application threads. A common general assumption is
    # using 2 per available processor cores - to handle
    # incoming requests using one and performing background
    # operations using the other.
    THREADS_PER_PAGE = 2

    # Use a secure, unique and absolutely secret key for
    # signing the data.
    CSRF_SESSION_KEY = "secret"