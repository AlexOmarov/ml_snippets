"""
Neural network for generation of sound

This script allows user to generate some random music sound effects.

This tool accepts audio file as a parameter.

This script requires that `tensorflow` be installed within the Python environment.

This file can also be imported as a module.
Public interface:

    * train - trains model, set global model, transforms to tensorflow lite and saves on a drive
    * generate - generate random sound
"""

from tensorflow import keras
from werkzeug.datastructures import FileStorage

from business.util.ml_logger import logger
from presentation.api.audio_generation_result import AudioGenerationResult
from presentation.api.train_result import TrainResult

_global_model: keras.models.Sequential
_log = logger.get_logger(__name__.replace('__', '\''))

_MODEL_TFLITE_NAME = 'audio/generation/model.tflite'
_MODEL_PLOT_NAME = 'audio/generation/model.png'
_MODEL_NAME = "audio/generation/audio"


def generate(audio: FileStorage) -> AudioGenerationResult:
    """
    Makes a generation based on passed audio.

    :param audio: Storage with audio data
    """
    print(audio)
    return AudioGenerationResult("test23")


def train(metric: str) -> TrainResult:
    """
    Prepares model for generating.
    Creates model, updates global model, writes model to disk and converts in to tensorflow lite
    If the argument `metric` isn't passed in, the default accuracy metric is used.

    :param metric : The metric for model compilation (optional)

    :return TrainResult

    :exception NotImplementedError If passed metric isn't supported.
    """
    return TrainResult(metric=metric, data=[""])
