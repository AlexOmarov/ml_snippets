import os


class Config(object):
    # Определяет, включен ли режим отладки
    # В случае если включен, flask будет показывать
    # подробную отладочную информацию. Если выключен -
    # - 500 ошибку без какой-либо дополнительной информации.
    DEBUG = False
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    FILE_ENCODING = "utf-8"
    SERVER_PORT = 5000
    CSRF_ENABLED = True
    # Application threads. A common general assumption is
    # using 2 per available processor cores - to handle
    # incoming requests using one and performing background
    # operations using the other.
    THREADS_PER_PAGE = 2
    # Use a secure, unique and absolutely secret key for signing the data.
    CSRF_SESSION_KEY = "secret"
    MODELS_DIR_PATH = os.path.join(ROOT_DIR, "data/models/")

    # AUDIO GENERATION SETTINGS
    AG_LEARNING_RATE = 0.001
    AG_VALIDATION_SPLIT = 0.2
    AG_SAMPLE_RATE = 16000
    AG_BATCH_SIZE = 64
    AG_NUM_EPOCHS = 50
    AG_STEPS_PER_EPOCH = 40
    AG_NUM_MELS = 64  # number of mel spectrogram bins
    AG_VOCAB_SIZE = 50
    AG_HOP_LENGTH = 512
    AG_FRAME_LENGTH = 2048

    AG_WORDS_REGEX = r"\b\w+\b"
    AG_MODEL_NAME = "RUSSIAN_TTS_MODEL"
    AG_MODEL_LANG = "ru"
    AG_CHECKPOINT_FILE_NAME_TEMPLATE = "model.{epoch:02d}.h5"
    AG_CHECKPOINT_DIR_PATH = os.path.join(ROOT_DIR, "data/checkpoints/")
    AG_CHECKPOINT_FILE_PATH_TEMPLATE = os.path.join(AG_CHECKPOINT_DIR_PATH, AG_CHECKPOINT_FILE_NAME_TEMPLATE)
    AG_OUTPUT_FILE_PATH = os.path.join(ROOT_DIR, "data/output.wav")
    AG_SERIALIZED_UNITS_DIR_PATH = "F:/pickles"
    AG_TEST_SERIALIZED_UNITS_DIR_PATH = "F:/test_pickles"
    AG = os.path.join(MODELS_DIR_PATH, "audio_model.h5")
    AG_AUDIO_DIR_PATH = "D:/audio_dataset/audio_files"
    AG_INPUT_DIR_PATH = os.path.join(ROOT_DIR, "data/ogg/")
    AG_METADATA_FILE_PATH = "D:/audio_dataset/df.csv"
    AG_SPEAKER_FILE_PATH = "D:/audio_dataset/speakers.csv"
    AG_PHONEMES_FILE_PATH = os.path.join(ROOT_DIR, "data/phonemes.txt")
