import tensorflow as tf
from tensorflow.python.client.session import Session
from tensorflow.python.platform.tf_logging import log
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

v1 = tf.compat.v1


class CustomModel:
    session: Session
    scope_name: str

    def __init__(self, session, scope_name):
        print(self)
        self.session = session
        self.scope_name = scope_name

    def inference(self, x):
        # here x has 784 columns and N rows where N = size of minibatch
        with self.session:
            with v1.variable_scope("hidden_1") as scope:
                hidden_1 = self._layer(x, [784, 256], [256])
                scope.reuse_variables()
            with v1.variable_scope("hidden_2") as scope:
                hidden_2 = self._layer(hidden_1, [256, 256], [256])
                scope.reuse_variables()
            with v1.variable_scope("output") as scope:
                output = self._layer(hidden_2, [256, 10], [10])
                scope.reuse_variables()
            return output
        # here x has 10 columns and N rows where N = size of minibatch

    def loss(self, output, y):
        with self.session:
            # For each result in minibatch count the error
            dot_product = y * log(output)
            # Reduction along axis 0 collapses each column into a
            # single value, whereas reduction along axis 1 collapses
            # each row into a single value. In general, reduction along
            # axis i collapses the ith dimension of a tensor to size 1.
            xentropy = -tf.reduce_sum(dot_product, reduction_indices=1)
            loss = tf.reduce_mean(xentropy)
            return loss

    def training(self, loss, global_step, learning_rate):
        with self.session:
            v1.scalar_mul("loss", loss)
            optimizer = GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)
            return train_op

    def evaluate(self, output, y):
        with self.session:
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return accuracy

    def _layer(self, inp, weight_shape, bias_shape):
        with self.session:
            weight_stddev = (2.0 / weight_shape[0]) ** 0.5
            w_init = tf.random_normal_initializer(stddev=weight_stddev)
            bias_init = tf.constant_initializer(value=0)
            W = v1.get_variable("W", weight_shape, initializer=w_init)
            b = v1.get_variable("b", bias_shape, initializer=bias_init)
            return tf.nn.relu(tf.matmul(inp, W) + b)
