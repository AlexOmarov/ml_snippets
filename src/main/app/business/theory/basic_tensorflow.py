"""

"""

import tensorflow as tf
from flask import Request

from presentation.api.theory_result import TheoryResult

v1 = tf.compat.v1
sess = v1.Session(config=v1.ConfigProto(log_device_placement=True, allow_soft_placement=True))


def first(request: Request) -> TheoryResult:
    """
    :param
    """
    with v1.variable_scope("shared_variables") as scope, tf.device('/gpu:3'):
        v1.disable_eager_execution()
        first_input = v1.placeholder(tf.float32, [1000, 784], name="first_input")
        _my_network(first_input)
        scope.reuse_variables()
        second_input = v1.placeholder(tf.float32, [1000, 784], name="second_input")
        _my_network(second_input)

    return TheoryResult("Request: " + request.__str__())


def _my_network(inp):
    with v1.variable_scope("layer_1"):
        output_1 = _layer(inp, [784, 100], [100])
    with v1.variable_scope("layer_2"):
        output_2 = _layer(output_1, [100, 50], [50])
    with v1.variable_scope("layer_3"):
        output_3 = _layer(output_2, [50, 10], [10])
    return output_3


def _layer(inp, weight_shape, bias_shape):
    weight_init = tf.random_uniform_initializer(minval=-1, maxval=1)
    bias_init = tf.constant_initializer(value=0)
    matrix_w = v1.get_variable("W", weight_shape, initializer=weight_init)
    b = v1.get_variable("b", bias_shape, initializer=bias_init)
    return tf.matmul(inp, matrix_w) + b
