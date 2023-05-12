"""
Training of neural network for speech generation

This script allows user to train model based on audio files and it's metadata to generate speech.
Speech can be of various languages and voices

This script accepts training params setting as a parameter.

This script requires that all the requirements from the <OS>_requirements.txt are installed in python env.

This file can also be imported as a module.
Public interface:

    * train - trains model, transforms to tensorflow lite and saves on a drive
"""

import tensorflow as tf
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from business.audio.generation.dto.training_dataset import TrainingDataset
from business.audio.generation.dto.training_hyper_params_info import TrainingHyperParamsInfo
from business.audio.generation.dto.training_setting import TrainingSetting
from business.audio.generation.dto.training_unit import TrainingUnit
from presentation.api.train_result import TrainResult


def train(setting: TrainingSetting) -> TrainResult:

    normalized_dataset = _get_dataset(_get_training_units(setting))

    model = _get_model(setting.hyper_params_info)

    model.fit(
        x=[normalized_dataset.training_data, normalized_dataset.training_responses],
        y=[normalized_dataset.test_data, normalized_dataset.test_responses],
        batch_size=setting.hyper_params_info.batch_size,
        epochs=setting.hyper_params_info.num_epochs,
        validation_split=0.2,
        callbacks=[ModelCheckpoint(filepath=setting.paths_info.checkpoint_path_template)]
    )

    path = model.save(setting.paths_info.model_dir_path)
    tflite_path = _convert_to_lite(model, setting)

    return TrainResult(metric="", path=path, tflite_path=tflite_path, data=[])


def _convert_to_lite(model: Model, setting: TrainingSetting) -> str:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_file_path = setting.paths_info.model_dir_path + "/" + setting.model_name + ".tflite"
    with open(tflite_file_path, 'wb') as f:
        f.write(tflite_model)
    return tflite_file_path


def _get_training_units(setting: TrainingSetting) -> [TrainingUnit]:
    return []


def _get_dataset(units: [TrainingUnit]) -> TrainingDataset:
    dataset = _form_dataset(units)
    return _normalize_dataset(dataset)


def _form_dataset(units: [TrainingUnit]) -> TrainingDataset:
    return TrainingDataset()


def _normalize_dataset(dataset: TrainingDataset) -> TrainingDataset:
    return dataset


def _get_model(hyper_params: TrainingHyperParamsInfo) -> tf.keras.models.Model:
    model = Model(inputs=[inputs, text_inputs], outputs=[decoder_output, postnet_output])
    loss_fun = hyper_params.loss_fun  # MeanSquaredError()
    model.compile(optimizer=Adam(learning_rate=hyper_params.learning_rate), loss=[loss_fun, loss_fun])
    model.summary()
    return model


train(
    TrainingSetting(

    )
)
