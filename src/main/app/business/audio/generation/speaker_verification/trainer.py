import os
import pickle

import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam

from business.audio.generation.dto.training_setting import TrainingSetting
from business.audio.generation.dto.audio_entry import AudioEntry
from business.util.ml_logger import logger
from src.main.resource.config import Config

_log = logger.get_logger(__name__.replace('__', '\''))


def train(setting: TrainingSetting):
    batches_amount = len(os.listdir(setting.paths_info.serialized_units_dir_path))

    model = _get_speaker_encoder(setting)
    generator = _get_dataset_generator(setting)
    model.fit_generator(
        generator=generator,
        steps_per_epoch=batches_amount,
        epochs=setting.hyper_params_info.num_epochs,
        shuffle=True
    )
    tf.keras.utils.plot_model(model, Config.MODEL_DIR_PATH + "speaker_verification_plot", show_shapes=True)
    model.save(Config.MODEL_DIR_PATH + "speaker_verification")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(Config.MODEL_DIR_PATH + "speaker_verification_tf_lite", 'wb') as f:
        f.write(tflite_model)


def _get_speaker_encoder(setting: TrainingSetting) -> tf.keras.models.Model:
    # TODO: get speaker model
    model = Model(inputs=[mfcc_input, phoneme_input], outputs=output)
    loss_fun = setting.hyper_params_info.loss_fun  # MeanSquaredError()
    model.compile(optimizer=Adam(learning_rate=setting.hyper_params_info.learning_rate), loss=[loss_fun, loss_fun])
    model.summary()
    return model


def _get_dataset_generator(setting: TrainingSetting):
    batch_number = 1
    batches_amount = len(os.listdir(setting.paths_info.serialized_units_dir_path))
    while batches_amount >= batch_number:
        filename = f"/serialized_batch_{batch_number}.pkl"
        with open(setting.paths_info.serialized_units_dir_path + filename, 'rb') as file:
            units: [AudioEntry] = pickle.load(file)
        batch_features = [unit.feature_vector for unit in units]
        batch_identification_vectors = [unit.speaker_identification_vector for unit in units]
        yield batch_features, batch_identification_vectors
        batch_number += 1
