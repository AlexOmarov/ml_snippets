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
#  Lib imports
from tensorflow import keras as keras

#  App imports
from business.util.ml_logger import logger
from business.util.ml_tensorboard import histogram_callback

log = logger.get_logger(__name__.replace('__', '\''))


# Private functions

def _get_probability_model(model: keras.Sequential) -> keras.Sequential:
    # Layer which has same amount of neurons as in previous layer and applies softmax alg for each neuron activation
    # Sum of the neuron outputs = 1? each output in [0,1]
    return keras.Sequential([model, keras.layers.Softmax()])


def _fit(model: keras.Sequential, x_train, y_train, callbacks: list[keras.callbacks.TensorBoard]) -> None:
    # Train model with number of epochs
    model.fit(x_train, y_train, epochs=5, callbacks=callbacks)


def _compile_model(model: keras.Sequential, loss_fn, metric: str) -> None:
    # Adding loss function, metric and optimizer to model
    # Optimizer - algorithm which will be used for going through neurons and weights and changing weights
    model.compile(optimizer='adam', loss=loss_fn, metrics=[metric])


def _get_loss_function() -> keras.losses.SparseCategoricalCrossentropy:
    # Function which defines losses after each optimization loop (related to passed metric)
    return keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def _get_model() -> keras.models.Sequential:
    # Create simple sequential model (each layer after another). Model - collection of layers
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # Flat incoming 28x28 matrix to single vector
        keras.layers.Dense(128, activation='relu'),  # Fully integrated within previous layer
        # Delete neurons from previous layer with probability 0.2 (make it less overfitting, more sparsed)
        keras.layers.Dropout(0.2),
        # Last output layer, which has 10 elements (one by each category)
        keras.layers.Dense(10)
    ])
    log.info(str.format("Got model {0}", model))
    return model


def _get_dataset() -> tuple:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize image vectors (make values interval less)
    return (x_train, y_train), (x_test, y_test)


def predict(image: str):
    # TODO: Process prediction
    result = _global_model.predict(image)
    return ""


def train(metric: str):
    """
    Prepares model for predictions.

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
    global _global_model
    # Get basic vars
    (x_train, y_train), (x_test, y_test) = _get_dataset()  # x - images, y - labels
    model = _get_model()
    loss_fn = _get_loss_function()

    # Train model
    _compile_model(model, loss_fn, metric)
    _fit(model, x_train, y_train, [histogram_callback.get_histogram_callback(1)])
    result = model.evaluate(x_test, y_test, verbose=2)

    # Create probability model
    probability_model = _get_probability_model(model)

    # Get final tensor
    print(probability_model(x_test[:5]))
    # Print model scheme
    keras.utils.plot_model(model, "models/model.png", show_shapes=True)
    model.save("models/model")
    _global_model = model
    return result


_global_model: keras.models.Sequential = train('accuracy')
