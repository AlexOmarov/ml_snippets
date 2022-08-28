# This is a Python script for tensorflow start.

import tensorflow as tf


def print_hi():
    print("TensorFlow version:", tf.__version__)
    print(tf.config.list_physical_devices('GPU'))


if __name__ == '__main__':
    print_hi()
