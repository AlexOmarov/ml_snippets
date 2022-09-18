"""
Script for classification of audio files using spectrogram and CNN

This script allows user classify audio file into several genres.

This tool accepts no parameters.

This script requires that `tensorflow` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following functions:

    * train - trains model, set global model, transforms to tensorflow lite and saves on a drive
    * predict - gets an audio and returns ClassificationResult with predicted class
"""

#  Lib imports
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, AveragePooling2D, Activation, Dropout, Flatten, Dense, Softmax
from tensorflow import keras as keras
from tensorflow.python.framework.ops import EagerTensor
from werkzeug.datastructures import FileStorage

#  App imports
from business.util.ml_logger import logger
from business.util.ml_tensorboard import histogram_callback
from presentation.api.classification_result import ClassificationResult
from presentation.api.train_result import TrainResult
from src.resource.config import Config

_global_model: keras.models.Sequential
_log = logger.get_logger(__name__.replace('__', '\''))

_ERROR_LABEL = "NOT_DEFINED"
_GENRES = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()


def predict(audio: FileStorage) -> ClassificationResult:
    """
    Makes a prediction based on passed image.

    Parameters
    ----------
    audio : FileStorage
             storage with image info
    """
    result = _make_prediction(_preprocess_single_audio(audio.read()))
    return ClassificationResult(imageName=audio.filename, label=result)


def train(metric: str):
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
    model = _get_model()
    loss_fn = _get_loss_function()

    # Create probability model
    probability_model = _get_probability_model(model)

    # Train model
    _compile_model(probability_model, loss_fn, metric)
    _fit(probability_model, x_train, y_train, [histogram_callback.get_histogram_callback(1)])
    result = probability_model.evaluate(x_test, y_test, verbose=2)

    # Get final tensor
    print(probability_model(x_test[:5]))

    # Refresh model sources
    _refresh_model_sources(probability_model)

    # Convert to Tensorflow lite
    _convert_to_lite(probability_model)

    return TrainResult(metric=metric, data=result)


# Private functions

def _get_probability_model(model: keras.Sequential) -> Sequential:
    # Layer which has same amount of neurons as in previous layer and applies softmax alg for each neuron activation
    # Sum of the neuron outputs = 1? each output in [0,1]
    model.add(Softmax())
    return model


def _fit(model: keras.Sequential, x_train, y_train, callbacks: list[keras.callbacks.TensorBoard]) -> None:
    # Train model with number of epochs
    model.fit(
        x_train, validation_data=y_train, epochs=50, steps_per_epoch=100, callbacks=callbacks, validation_steps=200
    )


def _compile_model(model: keras.Sequential, loss_fn, metric: str) -> None:
    # Adding loss function, metric and optimizer to model
    # Optimizer - algorithm which will be used for going through neurons and weights and changing weights
    model.compile(optimizer='sgd', loss=loss_fn, metrics=[metric])


def _get_loss_function() -> keras.losses.SparseCategoricalCrossentropy:
    # Function which defines losses after each optimization loop (related to passed metric)
    return keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def _get_model() -> Sequential:
    # Create simple sequential model (each layer after another). Model - collection of layers
    model = Sequential()
    # first hidden layer
    model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=(64, 64, 3)))
    model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(Activation('relu'))
    # second hidden layer
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(Activation('relu'))
    # third hidden layer
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(Activation('relu'))
    # flatten layer
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    # dense layer
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    # output layer
    model.add(Dense(10))
    return model


def _get_dataset() -> tuple:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize image vectors (make values interval less)
    return (x_train, y_train), (x_test, y_test)


def _convert_to_lite(model: Sequential):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    # Save the model.
    with open(Config.MODEL_PATH + 'model.tflite', 'wb') as f:
        f.write(tflite_model)


def _refresh_model_sources(model: Sequential):
    # Refresh model sources
    global _global_model
    keras.utils.plot_model(model, Config.MODEL_PATH + "model.png", show_shapes=True)
    model.save(Config.MODEL_PATH + "model")
    _global_model = model


def _make_prediction(tensor: EagerTensor) -> str:
    result = _ERROR_LABEL
    if '_global_model' in globals():
        predictions = _global_model.predict(tensor)
        result = np.argmax(predictions, axis=1)[0].__str__()
    return result


def _preprocess_single_audio(audio_bytes):
    image = tf.audio.decode_wav(audio_bytes, channels=1)
    return image
