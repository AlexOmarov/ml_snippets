import tensorflow as tf
from tensorflow.python.platform.tf_logging import log
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

v1 = tf.compat.v1


class CustomModel:
    def __init__(self):
        print(self)

    def inference(self, x):
        init = tf.constant_initializer(value=0)
        matrix_w = v1.get_variable("W", [784, 10], initializer=init)
        b = v1.get_variable("b", [10], initializer=init)
        output = tf.nn.softmax(tf.matmul(x, matrix_w) + b)
        return output

    def loss(self, output, y):
        dot_product = y * log(output)
        # Reduction along axis 0 collapses each column into a
        # single value, whereas reduction along axis 1 collapses
        # each row into a single value. In general, reduction along
        # axis i collapses the ith dimension of a tensor to size 1.
        xentropy = -tf.reduce_sum(dot_product, reduction_indices=1)
        loss = tf.reduce_mean(xentropy)
        return loss

    def training(self, cost, global_step, learning_rate):
        optimizer = GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(cost,
                                      global_step=global_step)
        return train_op

    def evaluate(self, output, y):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
