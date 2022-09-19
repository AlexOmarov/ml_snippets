"""
Sequential neural network for classifying MNIST digits dataset

This script allows user to classify images from mnist dataset or similar.

This tool accepts no parameters.

This script requires that `tensorflow` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following functions:

    * train - trains model, set global model, transforms to tensorflow lite and saves on a drive
    * predict - gets an image and returns ClassificationResult with predicted class
"""

#  Lib imports
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.losses import SparseCategoricalCrossentropy, Loss
from numpy import ndarray
from tensorflow import keras as keras
from tensorflow.python.framework.ops import EagerTensor
from werkzeug.datastructures import FileStorage

#  App imports
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

    Parameters
    ----------
    image : FileStorage
             storage with image info
    """
    result: str = _make_prediction(_preprocess_single_image(image.read()))
    return ClassificationResult(image_name=image.filename, label=result)


def train(metric: str) -> TrainResult:
    """
    Prepares model for predictions.
    Creates model, updates global model, writes model to disk and converts in to tensorflow lite

    If the argument `metric` isn't passed in, the default accuracy metric is used.

    Parameters
    ----------
    metric : str, optional
             The metric for model compilation

    Raises
    ------
    NotImplementedError
        If passed metric isn't supported.
    """
    # Get basic vars
    (x_train, y_train), (x_test, y_test) = _get_dataset()  # x - images, y - labels
    model: Sequential = _get_model()
    loss_fn: Loss = _get_loss_function()

    # Train model
    _compile_model(model, loss_fn, metric)
    _fit(model, x_train, y_train, [histogram_callback.get_histogram_callback(1)])
    result: list = model.evaluate(x_test, y_test, verbose=2)

    # Create probability model
    probability_model = _get_probability_model(model)

    # Refresh model sources
    _refresh_model_sources(probability_model)

    # Convert to Tensorflow lite
    _convert_to_lite(probability_model)

    return TrainResult(metric=metric, data=result)


# Private functions

def _get_probability_model(model: Sequential) -> Sequential:
    # Layer which has same amount of neurons as in previous layer and applies softmax alg for each neuron activation
    # Sum of the neuron outputs = 1? each output in [0,1]
    # TODO: not [0,1], why?
    model.add(keras.layers.Softmax())
    return model


def _fit(model: keras.Sequential, x_train, y_train, callbacks: list[keras.callbacks.TensorBoard]) -> None:
    # Train model with number of epochs
    model.fit(x_train, y_train, epochs=5, callbacks=callbacks)


def _compile_model(model: keras.Sequential, loss_fn, metric: str) -> None:
    # Adding loss function, metric and optimizer to model
    # Optimizer - algorithm which will be used for going through neurons and weights and changing weights
    model.compile(optimizer="adam", loss=loss_fn, metrics=[metric])


def _get_loss_function() -> Loss:
    # Function which defines losses after each optimization loop (related to passed metric)
    return SparseCategoricalCrossentropy(from_logits=True)


def _get_model() -> Sequential:
    # Create simple sequential model (each layer after another). Model - collection of layers
    model: Sequential = Sequential([
        Flatten(input_shape=_IMAGE_SIZE),  # Flat incoming 28x28 matrix to single vector
        Dense(128, activation="relu"),  # Fully integrated within previous layer
        # Delete neurons from previous layer with probability 0.2 (make it less over-trained, more sparse)
        Dropout(0.2),
        # Last output layer, which has 10 elements (one by each category)
        Dense(10)
    ])
    return model


def _get_dataset() -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Normalize image vectors (make values interval less)
    x_train, x_test = x_train / _NORMALIZATION, x_test / _NORMALIZATION
    return (x_train, y_train), (x_test, y_test)


def _convert_to_lite(model: keras.models.Sequential) -> None:
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    # Save the model.
    with open(Config.MODEL_PATH + _MODEL_TFLITE_NAME, 'wb') as f:
        f.write(tflite_model)


def _refresh_model_sources(model: keras.models.Sequential) -> None:
    # Refresh model sources
    global _global_model
    keras.utils.plot_model(model, Config.MODEL_PATH + _MODEL_PLOT_NAME, show_shapes=True)
    model.save(Config.MODEL_PATH + _MODEL_NAME)
    _global_model = model


def _make_prediction(tensor: EagerTensor) -> str:
    result = _ERROR_LABEL
    if '_global_model' in globals():
        predictions = _global_model.predict(tensor)
        result = np.argmax(predictions, axis=1)[0].__str__()
    return result


def _preprocess_single_image(image_bytes) -> EagerTensor:
    image = tf.image.decode_jpeg(image_bytes, channels=1)
    image = tf.image.resize(image, size=_IMAGE_SIZE)
    image: EagerTensor = tf.expand_dims(image[:, :, 0], 0)
    image = image / _NORMALIZATION  # Normalize final tensor
    return image
