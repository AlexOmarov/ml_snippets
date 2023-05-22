import os

from keras.losses import MeanSquaredError


class Config(object):
    # Определяет, включен ли режим отладки
    # В случае если включен, flask будет показывать
    # подробную отладочную информацию. Если выключен -
    # - 500 ошибку без какой-либо дополнительной информации.
    DEBUG = False
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    MODEL_DIR_PATH = os.path.join(ROOT_DIR, "data/models/")
    AUDIO_MODEL_PATH = os.path.join(ROOT_DIR, "data/models/audio_model.h5")
    AUDIO_FILES_DIR_PATH = "D:/audio_dataset/audio_files"
    AUDIO_CONVERSION_INPUT_DIR_PATH = os.path.join(ROOT_DIR, "data/ogg/")
    METADATA_FILE_PATH = "D:/audio_dataset/df.csv"
    SPEAKER_FILE_PATH = "D:/audio_dataset/speakers.csv"
    PHONEMES_FILE_PATH = os.path.join(ROOT_DIR, "data/phonemes.txt")

    AUDIO_CONVERSION_OUTPUT_EXT = "wav"

    WORDS_FILE_ENCODING = "utf-8"
    WORDS_REGEX = r"\b\w+\b"
    RECOGNIZE_LANGUAGE = "ru-RU"
    PHONEMIZE_LANGUAGE = "ru"
    PHONEMIZE_BACKEND = "espeak"  # Install ESPEAK NG as msi from github releases

    AUDIO_GENERATION_LEARNING_RATE = 0.001
    AUDIO_GENERATION_CHECKPOINT_FILE_NAME_TEMPLATE = "model.{epoch:02d}.h5"
    AUDIO_GENERATION_SAMPLE_RATE = 22050
    AUDIO_GENERATION_BATCH_SIZE = 64
    AUDIO_GENERATION_NUM_EPOCHS = 50
    AUDIO_GENERATION_STEPS_PER_EPOCH = 40
    AUDIO_GENERATION_ENCODER_LAYERS = 3
    AUDIO_GENERATION_DECODER_LAYERS = 3
    AUDIO_GENERATION_POST_KERNEL_SIZE = 4
    AUDIO_GENERATION_VALIDATION_SPLIT = 0.2
    AUDIO_GENERATION_MODEL_NAME = "RUSSIAN_TTS_MODEL"
    AUDIO_GENERATION_LOSS_FUN = MeanSquaredError()
    AUDIO_GENERATION_NUM_MELS = 64  # number of mel spectrogram bins
    AUDIO_GENERATION_VOCAB_SIZE = 80
    AUDIO_GENERATION_HOP_LENGTH = 512
    AUDIO_GENERATION_FRAME_LENGTH = 4056
    AUDIO_GENERATION_CHECKPOINT_DIR_PATH = os.path.join(ROOT_DIR, "data/checkpoints/")
    AUDIO_GENERATION_CHECKPOINT_FILE_PATH_TEMPLATE = os.path.join(AUDIO_GENERATION_CHECKPOINT_DIR_PATH,
                                                                  AUDIO_GENERATION_CHECKPOINT_FILE_NAME_TEMPLATE)
    AUDIO_GENERATION_OUTPUT_FILE_PATH = os.path.join(ROOT_DIR, "data/output.wav")
    AUDIO_GENERATION_PICKLED_UNITS_DIR_PATH = "F:/pickles"

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
