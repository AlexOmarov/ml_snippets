import tensorflow as tf
from keras import layers, Model
from keras.utils import plot_model
from tensorflow.lite.python.convert import OpsSet
from tensorflow.lite.python.lite import TFLiteConverter

from business.audio.generation.train.train import TrainingSetting
from business.audio.generation.train.train import get_dataset_generator
from business.util.ml_logger import logger

_log = logger.get_logger(__name__.replace('__', '\''))


def create_model(setting: TrainingSetting):
    model = _get_speaker_encoder()
    init_batch = [1]
    model.fit(
        x=get_dataset_generator(setting.paths_info.serialized_units_dir_path, init_batch),
        steps_per_epoch=setting.hyper_params_info.steps_per_epoch,
        epochs=setting.hyper_params_info.num_epochs,
        shuffle=True
    )
    model_file_path = setting.paths_info.models_dir_path + setting.model_name
    plot_model(model, to_file=model_file_path + ".png", show_shapes=True)
    model.save(model_file_path)
    converter = TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [OpsSet.TFLITE_BUILTINS, OpsSet.SELECT_TF_OPS]  # enable TF Lite and TF ops.
    tflite_model = converter.convert()

    with open(model_file_path + "_lite", 'wb') as f:
        f.write(tflite_model)


def _get_speaker_encoder() -> tf.keras.models.Model:
    input_shape = (149,)
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128)(inputs)
    outputs = layers.Dense(66, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()
    return model
