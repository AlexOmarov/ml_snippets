"""

"""

import tensorflow as tf
from flask import Request
from numpy import ndarray

from business.theory.model import CustomModel
from presentation.api.theory_result import TheoryResult

v1 = tf.compat.v1
sess = v1.Session(config=v1.ConfigProto(log_device_placement=True, allow_soft_placement=True))

_NORMALIZATION = 255.0


def first() -> TheoryResult:
    """
    :param
    """
    # Parameters
    global sess
    learning_rate = 0.01
    training_epochs = 1000
    batch_size = 100
    display_step = 1
    with tf.Graph().as_default():
        # mnist data image of shape 28*28=784
        x = v1.placeholder("float", [None, 784])
        # 0-9 digits recognition => 10 classes
        y = v1.placeholder("float", [None, 10])
        model = CustomModel(sess, "name")
        output = model.inference(x)
        cost = model.loss(output, y)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = model.training(cost, global_step, learning_rate)
        eval_op = model.evaluate(output, y)
        init_op = v1.initialize_all_variables()
        sess.run(init_op)
        dataset = _get_dataset()
        train = dataset[0]
        test = dataset[1]

        # Training cycle
        for epoch in range(training_epochs):
            total_batch = int(train[0].__sizeof__() / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                mbatch_x, mbatch_y = _get_batch(train, batch_size, i)
                # Fit training using batch data
                feed_dict = {x: mbatch_x, y: mbatch_y}
                sess.run(train_op, feed_dict=feed_dict)
                # Display logs per epoch step
                if epoch % display_step == 0:
                    val_feed_dict = {x: test[0], y: test[1]}
                accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
                print("VALIDATION ERROR: {}".format(1 - accuracy))

    print("Optimization Finished!")
    test_feed_dict = {x: test[0], y: test[1]}
    accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
    print("Test Accuracy:", accuracy)
    return TheoryResult("Request: ")


def _get_dataset() -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize image vectors (make values interval less)
    x_train, x_test = x_train / _NORMALIZATION, x_test / _NORMALIZATION
    return (x_train, y_train), (x_test, y_test)


def _get_batch(train: tuple[ndarray, ndarray], batch_size: int, batch_number: int) -> tuple[ndarray, ndarray]:
    start = batch_number * batch_size
    end = start + batch_size
    if start < train[0].__sizeof__():
        return train
    if end > train[0].__sizeof__():
        end = train[0].__sizeof__()
    return train[0][start:end], train[0][start:end]

# TODO: fix Session context managers are not re-entrant. Use `Session.as_default()` if you want to enter a session multiple times.
first()
