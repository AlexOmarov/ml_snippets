# This is a Python script for tensorflow start.
from tensorflow import keras


def train():
    print("Keras version:", keras.__version__)

    # Get basic vars
    (x_train, y_train), (x_test, y_test) = _get_dataset()  # x - images, y - labels
    model = _get_model()
    loss_fn = _get_loss_function()

    # Train model
    _compile_model(model, loss_fn)
    _fit(model, x_train, y_train)
    model.evaluate(x_test, y_test, verbose=2)

    # Create probability model
    probability_model = _get_probability_model(model)

    # Get final tensor
    print(probability_model(x_test[:5]))


# Private functions

def _get_probability_model(model):
    return keras.Sequential([model, keras.layers.Softmax()])


def _fit(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=5)


def _compile_model(model, loss_fn):
    model.compile(optimizer='adam', loss=loss_fn,  metrics=['accuracy'])


def _get_loss_function():
    return keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def _get_model():
    return keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])


def _get_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)
