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
import csv
import re
import sys

import librosa
import numpy as np
import psutil
import pymorphy2
import tensorflow as tf
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from numpy import ndarray
from phonemizer import phonemize
from pymorphy2 import MorphAnalyzer

from business.audio.generation.dto.training_dataset import TrainingDataset
from business.audio.generation.dto.training_hyper_params_info import TrainingHyperParamsInfo
from business.audio.generation.dto.training_paths_info import TrainingPathsInfo
from business.audio.generation.dto.training_setting import TrainingSetting
from business.audio.generation.dto.training_unit import TrainingUnit
from business.util.ml_logger import logger
from presentation.api.train_result import TrainResult
from src.main.resource.config import Config

_log = logger.get_logger(__name__.replace('__', '\''))


def train(setting: TrainingSetting) -> TrainResult:
    """
    Prepares model for generating.
    Creates model, writes model to disk and converts in to tensorflow lite

    :param setting : The settings for training

    :return TrainResult

    :exception RuntimeError If undefined exception happens.
    """

    normalized_dataset = _get_dataset(_get_training_units(setting))

    model = _get_model(setting.hyper_params_info)

    model.fit(
        x=[normalized_dataset.training_data, normalized_dataset.training_responses],
        y=[normalized_dataset.test_data, normalized_dataset.test_responses],
        batch_size=setting.hyper_params_info.batch_size,
        epochs=setting.hyper_params_info.num_epochs,
        validation_split=setting.hyper_params_info.validation_split,
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


def _get_dataset(units: [TrainingUnit]) -> TrainingDataset:
    dataset = _form_dataset(units)
    return _normalize_dataset(dataset)


def _get_training_units(setting: TrainingSetting) -> [TrainingUnit]:
    result: [TrainingUnit] = []
    morph = pymorphy2.MorphAnalyzer()
    mem = psutil.virtual_memory().available // 1024
    with open(setting.paths_info.metadata_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        index = 0
        for row in reader:
            unit = _form_training_unit(row, setting, morph)
            result.append(unit)
            array_size_kib = sys.getsizeof(result) / 1024
            unit_size_kib = sys.getsizeof(unit) / 1024
            percentage = array_size_kib * 100 // mem
            index = index + 1
            _log.info(
                "№ " + index.__str__() + "." +
                "Formed training unit from " + row[1].__str__() + "." +
                " Size of overall array: " + array_size_kib.__str__() + "KiB." +
                " Unit size: " + unit_size_kib.__str__() + " KiB." +
                " Memory size: " + mem.__str__() + " KiB." +
                " Memory usage: " + percentage.__str__() + " %." +
                " Unit " + unit.serialize().__str__()
            )

    return result


def _form_training_unit(row: list[str], setting: TrainingSetting, morph: MorphAnalyzer) -> TrainingUnit:
    sampling_rate = float(row[9])
    duration = float(row[10])
    audio_text = row[0]
    file_path = setting.paths_info.audio_files_dir_path + row[7].removeprefix("audio_files")
    audio = _get_audio(file_path, sampling_rate, duration)
    mfcc_db = _get_mfcc_db(audio, sampling_rate, setting)
    spectrogram = _get_spectrogram(audio, setting.frame_length, setting.hop_length)
    phonemes = _phonemize_text(audio_text, morph, setting.phonemize_language)
    return TrainingUnit(
        audio_path=file_path,
        text=row[0],
        phonemes=phonemes,
        sampling_rate=sampling_rate,
        duration_seconds=duration,
        mfcc_db=mfcc_db,
        spectrogram=spectrogram
    )


def _get_audio(path: str, sampling_rate: float, duration: float) -> ndarray:
    audio_data, sr = librosa.load(path, sr=sampling_rate, duration=duration, mono=True)

    # Normalize the audio
    audio_data /= max(abs(audio_data))

    # Trim leading and trailing silence
    audio_data, _ = librosa.effects.trim(audio_data)

    # Resample the audio if necessary
    if sr != sampling_rate:
        audio_data = librosa.resample(audio_data, sr, sampling_rate)
    return audio_data


def _phonemize_text(text: str, morph: MorphAnalyzer, language: str) -> list[str]:
    words = re.findall(Config.WORDS_REGEX, text)
    all_phonemes = []

    for word in words:
        base_form = morph.parse(word)[0].normal_form
        phonemes = phonemize(base_form, language=language)
        all_phonemes.append(phonemes)

    return all_phonemes


def _form_dataset(units: [TrainingUnit]) -> TrainingDataset:
    return TrainingDataset()


def _normalize_dataset(dataset: TrainingDataset) -> TrainingDataset:
    return dataset


def _get_mfcc_db(audio_data: ndarray, sampling_rate: float, setting: TrainingSetting) -> ndarray:
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate, n_mels=setting.num_mels, fmin=125,
                                              fmax=7600)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def _get_spectrogram(audio_data: ndarray, frame_length: int, hop_length: int):
    ft = librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.absolute(ft)
    return magnitude


def _get_model(hyper_params: TrainingHyperParamsInfo) -> tf.keras.models.Model:
    model = Model(inputs=[inputs, text_inputs], outputs=[decoder_output, postnet_output])
    loss_fun = hyper_params.loss_fun
    model.compile(optimizer=Adam(learning_rate=hyper_params.learning_rate), loss=[loss_fun, loss_fun])
    model.summary()
    return model


train(
    TrainingSetting(
        hyper_params_info=TrainingHyperParamsInfo(
            batch_size=Config.AUDIO_GENERATION_BATCH_SIZE,
            learning_rate=Config.AUDIO_GENERATION_LEARNING_RATE,
            num_epochs=Config.AUDIO_GENERATION_NUM_EPOCHS,
            loss_fun=Config.AUDIO_GENERATION_LOSS_FUN,
            validation_split=Config.AUDIO_GENERATION_VALIDATION_SPLIT,
            encoder_layers=Config.AUDIO_GENERATION_ENCODER_LAYERS,
            decoder_layers=Config.AUDIO_GENERATION_DECODER_LAYERS,
            post_kernel_size=Config.AUDIO_GENERATION_POST_KERNEL_SIZE
        ),
        paths_info=TrainingPathsInfo(
            metadata_file_path=Config.METADATA_FILE_PATH,
            phonemes_file_path=Config.PHONEMES_FILE_PATH,
            audio_files_dir_path=Config.AUDIO_FILES_DIR_PATH,
            checkpoint_path_template=Config.AUDIO_GENERATION_CHECKPOINT_FILE_PATH_TEMPLATE,
            model_dir_path=Config.MODEL_DIR_PATH
        ),
        model_name=Config.AUDIO_GENERATION_MODEL_NAME,
        num_mels=Config.AUDIO_GENERATION_NUM_MELS,
        frame_length=Config.AUDIO_GENERATION_FRAME_LENGTH,
        hop_length=Config.AUDIO_GENERATION_HOP_LENGTH,
        phonemize_language=Config.PHONEMIZE_LANGUAGE,
        vocab_size=Config.AUDIO_GENERATION_VOCAB_SIZE
    )
)
