"""
Neural network for classifying music

This script allows user to classify music into the corresponding genre.

This tool accepts audio file as a parameter.

This script requires that `tensorflow` be installed within the Python environment.

This file can also be imported as a module.
Public interface:

    * train - trains model, set global model, transforms to tensorflow lite and saves on a drive
    * predict - gets an audio and returns ClassificationResult with predicted genre
"""

from tensorflow import keras
from werkzeug.datastructures import FileStorage

from business.util.ml_logger import logger
from presentation.api.classification_result import ClassificationResult
from presentation.api.train_result import TrainResult

_global_model: keras.models.Sequential
_log = logger.get_logger(__name__.replace('__', '\''))

_MODEL_TFLITE_NAME = 'audio/model.tflite'
_MODEL_PLOT_NAME = 'audio/model.png'
_MODEL_NAME = "audio/audio"


def predict(audio: FileStorage) -> ClassificationResult:
    """
    Makes a prediction based on passed image.

    :param audio: Storage with audio data
    """
    return ClassificationResult(image_name=audio.filename, label="")


def train(metric: str) -> TrainResult:
    """
    Prepares model for predictions.
    Creates model, updates global model, writes model to disk and converts in to tensorflow lite
    If the argument `metric` isn't passed in, the default accuracy metric is used.

    :param metric : The metric for model compilation (optional)

    :return TrainResult

    :exception NotImplementedError If passed metric isn't supported.
    """

    return TrainResult(metric=metric, data=[""])
