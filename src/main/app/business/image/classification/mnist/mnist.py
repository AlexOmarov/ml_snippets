"""
Sequential neural network for classifying MNIST digits dataset

This script allows user to classify images from mnist dataset or similar.

This tool accepts no parameters.

This script requires that `tensorflow` be installed within the Python environment.

This file can also be imported as a module.
Public interface:

    * train - trains model, set global model, transforms to tensorflow lite and saves on a drive
    * predict - gets an image and returns ClassificationResult with predicted class
"""

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.losses import SparseCategoricalCrossentropy, Loss
from numpy import ndarray
from tensorflow import keras as keras
from tensorflow.python.framework.ops import EagerTensor
from werkzeug.datastructures import FileStorage

from business.util.ml_logger import logger
from business.util.ml_tensorboard import histogram_callback
from presentation.api.classification_result import ClassificationResult
from presentation.api.train_result import TrainResult
from src.main.resource.config import Config

_global_model: keras.models.Sequential
_log = logger.get_logger(__name__.replace('__', '\''))

_ERROR_LABEL = "NOT_DEFINED"

_IMAGE_SIZE = (28, 28)
_NORMALIZATION = 255.0

_MODEL_TFLITE_NAME = 'mnist/mnist.tflite'
_MODEL_PLOT_NAME = 'model.png'
_MODEL_NAME = "mnist/mnist"


def predict(image: FileStorage) -> ClassificationResult:
    """
    Makes a prediction based on passed image.

    :param image: Storage with image info
    """
    result: str = _make_prediction(_preprocess_single_image(image.read()))
    return ClassificationResult(image_name=image.filename, label=result)


def train(metric: str) -> TrainResult:
    """
    Prepares model for predictions.
    Creates model, updates global model, writes model to disk and converts in to tensorflow lite
    If the argument `metric` isn't passed in, the default accuracy metric is used.

    :param metric : The metric for model compilation (optional)

    :return TrainResult

    :exception NotImplementedError If passed metric isn't supported.
    """

    (train_images, train_labels), (test_images, test_labels) = _get_dataset()
    model: Sequential = _get_model()
    loss_fn: Loss = SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer="adam", loss=loss_fn, metrics=[metric])
    model.fit(train_images, train_labels, epochs=5, callbacks=[histogram_callback.get_histogram_callback(1)])
    result: list = model.evaluate(test_images, test_labels, verbose=2)

    probability_model = _get_probability_model(model)

    _refresh_model_sources(probability_model)

    _convert_to_lite(probability_model)

    return TrainResult(metric=metric, data=result)


def _make_prediction(tensor: EagerTensor) -> str:
    result = _ERROR_LABEL
    if '_global_model' in globals():
        predictions = _global_model.predict(tensor)
        result = np.argmax(predictions, axis=1)[0].__str__()
    return result


def _get_probability_model(model: Sequential) -> Sequential:
    model.add(keras.layers.Softmax())
    return model


def _get_model() -> Sequential:
    model: Sequential = Sequential([
        Flatten(input_shape=_IMAGE_SIZE),  # Flat incoming 28x28 matrix to single vector
        Dense(128, activation="relu"),
        Dropout(0.2),  # Delete neurons from previous layer with probability 0.2 (make it less over-trained)
        Dense(10)
    ])
    return model


def _get_dataset() -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Normalize image vectors (make values interval less)
    x_train, x_test = x_train / _NORMALIZATION, x_test / _NORMALIZATION
    return (x_train, y_train), (x_test, y_test)


def _convert_to_lite(model: keras.models.Sequential) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(Config.MODEL_PATH + _MODEL_TFLITE_NAME, 'wb') as f:
        f.write(tflite_model)


def _refresh_model_sources(model: keras.models.Sequential) -> None:
    global _global_model

    keras.utils.plot_model(model, Config.MODEL_PATH + _MODEL_PLOT_NAME, show_shapes=True)
    model.save(Config.MODEL_PATH + _MODEL_NAME)
    _global_model = model


def _preprocess_single_image(image_bytes) -> EagerTensor:
    image = tf.image.decode_jpeg(image_bytes, channels=1)
    image = tf.image.resize(image, size=_IMAGE_SIZE)
    image: EagerTensor = tf.expand_dims(image[:, :, 0], 0)
    image = image / _NORMALIZATION
    return image
