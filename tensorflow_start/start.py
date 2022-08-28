# This is a Python script for tensorflow start.

import tensorflow as tf


def get_probability_model(model):
    return tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])


def fit(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=5)


def compile_model(model, loss_fn):
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])


def get_loss_function():
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def get_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])


def get_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def train():
    print("TensorFlow version:", tf.__version__)
    print(tf.config.list_physical_devices('GPU'))

    # Get basic vars
    (x_train, y_train), (x_test, y_test) = get_dataset()
    model = get_model()
    loss_fn = get_loss_function()

    # Train model
    compile_model(model, loss_fn)
    fit(model, x_train, y_train)
    model.evaluate(x_test, y_test, verbose=2)

    # Create probability model
    probability_model = get_probability_model(model)

    # Get final tensor
    probability_model(x_test[:5])


if __name__ == '__main__':
    train()
