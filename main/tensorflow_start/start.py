"""
Sequential neural network for classifying MNIST dataset

This script allows the user to print to the console resulting tensor of sequential neural network for MNIST dataset.

This tool accepts no parameters.

This script requires that `tensorflow` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * train - prints the resulting tensor
"""

from tensorflow import keras as keras
from util import logger as log


def train(metric: str):
    """Prints resulting tensor of sequential NN for MNIST classification.

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
    log.info(str.format("Keras version: {0}", keras.__version__))

    # Get basic vars
    (x_train, y_train), (x_test, y_test) = _get_dataset()  # x - images, y - labels
    model = _get_model()
    loss_fn = _get_loss_function()

    # Train model
    _compile_model(model, loss_fn, metric)
    _fit(model, x_train, y_train)
    model.evaluate(x_test, y_test, verbose=2)

    # Create probability model
    probability_model = _get_probability_model(model)

    # Get final tensor
    print(probability_model(x_test[:5]))


# Private functions

def _get_probability_model(model: keras.Sequential) -> keras.Sequential:
    return keras.Sequential([model, keras.layers.Softmax()])


def _fit(model, x_train, y_train) -> None:
    model.fit(x_train, y_train, epochs=5)


def _compile_model(model, loss_fn, metric: str) -> None:
    model.compile(optimizer='adam', loss=loss_fn, metrics=[metric])


def _get_loss_function() -> keras.losses.SparseCategoricalCrossentropy:
    return keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def _get_model() -> keras.models.Sequential:
    # Create simple sequential model (each layer after another). Model - collection of layers
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # Flat incoming tensor to 28x28 matrix
        keras.layers.Dense(128, activation='relu'),  # Fully integrated
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    log.info(str.format("Got model {0}", model))
    return model


def _get_dataset() -> tuple:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)
