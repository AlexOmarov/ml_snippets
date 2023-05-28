import os
import pickle

import numpy as np
import tensorflow as tf
from keras import layers, Model

from business.audio.generation.dataset.dto.audio_entry import AudioEntry
from business.audio.generation.config.dto.training_setting import TrainingSetting
from business.util.ml_logger import logger
from src.main.resource.config import Config

_log = logger.get_logger(__name__.replace('__', '\''))


def train(setting: TrainingSetting):
    model = _get_speaker_encoder()
    init_batch = [1]
    model.fit(
        x=_get_dataset_generator(setting, init_batch),
        steps_per_epoch=42,
        epochs=setting.hyper_params_info.num_epochs,
        shuffle=True
    )
    (x_test, y_test) = _get_test_data(setting)
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    tf.keras.utils.plot_model(model, to_file=Config.MODEL_DIR_PATH + "speaker_verification_plot.png", show_shapes=True)
    model.save(Config.MODEL_DIR_PATH + "speaker_verification")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    with open(Config.MODEL_DIR_PATH + "speaker_verification_tf_lite", 'wb') as f:
        f.write(tflite_model)


def _get_test_data(setting) -> tuple:
    filename = "/serialized_batch_1.pkl"
    with open(setting.paths_info.serialized_units_dir_path + filename, 'rb') as file:
        units: [AudioEntry] = pickle.load(file)
    batch_features = [unit.feature_vector for unit in units]
    batch_identification_vectors = [unit.speaker_identification_vector for unit in units]
    return np.array(batch_features), np.array(batch_identification_vectors)


def _get_speaker_encoder() -> tf.keras.models.Model:
    input_shape = (149,)
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128)(inputs)
    outputs = layers.Dense(66, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()
    return model


def _get_dataset_generator(setting: TrainingSetting, init_batch: list):
    batches_amount = len(os.listdir(setting.paths_info.serialized_units_dir_path))
    while True:
        if batches_amount <= init_batch[0]:
            init_batch[0] = 4
        filename = f"/serialized_batch_{init_batch[0]}.pkl"
        with open(setting.paths_info.serialized_units_dir_path + filename, 'rb') as file:
            units: [AudioEntry] = pickle.load(file)
        batch_features = [unit.feature_vector for unit in units]
        batch_identification_vectors = [unit.speaker_identification_vector for unit in units]
        init_batch[0] = init_batch[0] + 1
        yield np.array(batch_features), np.array(batch_identification_vectors)
