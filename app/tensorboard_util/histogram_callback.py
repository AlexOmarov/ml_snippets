import datetime

from tensorflow import keras

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_histogram_callback(histogram_freq: int):
    return keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq)
