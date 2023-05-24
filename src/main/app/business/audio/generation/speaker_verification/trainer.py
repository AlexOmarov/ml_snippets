import csv
import os
import pickle

import numpy as np
import tensorflow as tf
from keras import layers, Model, Sequential
from numpy import ndarray

from business.audio.generation.config.training_setting import ts
from business.audio.generation.dto.audio_entry import AudioEntry
from business.audio.generation.dto.training_setting import TrainingSetting
from business.util.ml_logger import logger
from src.main.resource.config import Config

_log = logger.get_logger(__name__.replace('__', '\''))


def train(setting: TrainingSetting):
    model = _get_speaker_encoder(setting)
    generator = _get_dataset_generator(setting)
    model.fit(
        x=generator,
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
    result_features = []
    result_identifications = []
    for batch_number in range(3):
        filename = f"/serialized_batch_{batch_number + 1}.pkl"
        with open(setting.paths_info.serialized_units_dir_path + filename, 'rb') as file:
            units: [AudioEntry] = pickle.load(file)
        batch_features = [unit.feature_vector for unit in units]
        batch_identification_vectors = [unit.speaker_identification_vector for unit in units]
        result_features.append(batch_features)
        result_identifications.append(batch_identification_vectors)
        batch_number += 1
    return result_features, result_identifications


def _get_speaker_encoder(setting: TrainingSetting) -> tf.keras.models.Model:
    input_shape = (6 + 7 + setting.num_mels + 12 + 20 + 20 + 20,)
    speakers = _get_speakers(setting.paths_info.speaker_file_path)
    inputs = layers.Input(shape=input_shape)

    # Input embedding
    x = layers.Dense(128)(inputs)

    # Transformer encoder
    # Transformer encoder
    for _ in range(4):
        # Multi-head attention
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=128)
        x = attention(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward network
        ffn = Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(128),
        ])
        x = ffn(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Embedding layer
    x = layers.Dense(20 * 7)(x)
    x = layers.Reshape((20, 7))(x)

    # Output layer
    outputs = layers.Dense(66, activation="softmax", axis=-1)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def _get_speakers(path: str) -> ndarray:
    speakers = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        row = _next_row(reader)
        while row:
            speakers.append(row[0])
            row = _next_row(reader)
    return np.array(speakers)


def _next_row(reader) -> list[str]:
    try:
        return next(reader)
    except StopIteration:
        return []


def _get_dataset_generator(setting: TrainingSetting):
    batch_number = 4
    batches_amount = len(os.listdir(setting.paths_info.serialized_units_dir_path))
    while True:
        if batches_amount >= batch_number:
            batch_number = 4
        filename = f"/serialized_batch_{batch_number}.pkl"
        with open(setting.paths_info.serialized_units_dir_path + filename, 'rb') as file:
            units: [AudioEntry] = pickle.load(file)
        batch_features = [unit.feature_vector for unit in units]
        batch_identification_vectors = [unit.speaker_identification_vector for unit in units]
        yield np.array(batch_features), np.array(batch_identification_vectors)
        batch_number += 1


train(ts)