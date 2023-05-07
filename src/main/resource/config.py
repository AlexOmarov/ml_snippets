import os


class Config(object):
    # Определяет, включен ли режим отладки
    # В случае если включен, flask будет показывать
    # подробную отладочную информацию. Если выключен -
    # - 500 ошибку без какой-либо дополнительной информации.
    DEBUG = False
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    MODEL_PATH = "data/models/"
    AUDIO_FILES_DIR_PATH = os.path.join(ROOT_DIR, "data/wavs/")
    AUDIO_CONVERSION_INPUT_DIR = os.path.join(ROOT_DIR, "data/ogg/")
    AUDIO_CONVERSION_OUTPUT_EXT = "wav"
    WORDS_FILE_PATH = os.path.join(ROOT_DIR, "data/words.txt")
    WORDS_FILE_ENCODING = "utf-8"
    PHONEMES_FILE_PATH = os.path.join(ROOT_DIR, "data/phonemes.txt")
    RECOGNIZE_LANGUAGE = "ru-RU"
    PHONEMIZE_LANGUAGE = "ru"
    PHONEMIZE_BACKEND = "espeak"  # Install ESPEAK NG as msi from github releases

    AUDIO_GENERATION_LEARNING_RATE = 0.001
    AUDIO_GENERATION_BATCH_SIZE = 16
    AUDIO_GENERATION_NUM_EPOCHS = 50
    AUDIO_GENERATION_NUM_MELS = 80  # number of mel spectrogram bins
    AUDIO_GENERATION_VOCAB_SIZE = 80
    AUDIO_GENERATION_CHECKPOINT_DIR_PATH = os.path.join(ROOT_DIR, "data/checkpoints/")

    SERVER_PORT = 5000
    CSRF_ENABLED = True
    # Случайный ключ, которые будет исползоваться для подписи данных, например cookies.
    SECRET_KEY = 'YOUR_RANDOM_SECRET_KEY'

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # Application threads. A common general assumption is
    # using 2 per available processor cores - to handle
    # incoming requests using one and performing background
    # operations using the other.
    THREADS_PER_PAGE = 2

    # Use a secure, unique and absolutely secret key for signing the data.
    CSRF_SESSION_KEY = "secret"
