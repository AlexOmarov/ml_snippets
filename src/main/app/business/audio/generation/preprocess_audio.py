import csv
import os
import pickle
import re

import librosa
import numpy as np
import pymorphy2
from numpy import ndarray
from phonemizer import phonemize
from pymorphy2 import MorphAnalyzer

from business.audio.generation.dto.training_setting import TrainingSetting
from business.audio.generation.dto.training_unit import TrainingUnit
from business.audio.generation.config.training_setting import ts
from business.util.ml_logger import logger
from presentation.api.preprocess_result import PreprocessResult
from src.main.resource.config import Config

_log = logger.get_logger(__name__.replace('__', '\''))


def preprocess_audio(setting: TrainingSetting) -> PreprocessResult:
    """
    Prepares dataset based on .wav files, metafile .csv and config.

    :param setting : The settings for preprocessing

    :return TrainResult

    :exception RuntimeError If undefined exception happens.
    """

    paths = []
    morph = pymorphy2.MorphAnalyzer()
    # Get overall_processed_unit_amount (batch size should be same between calls)
    serialized_files = os.listdir(setting.paths_info.serialized_units_dir_path)
    last_serialized_file_number = len(serialized_files)
    overall_processed_unit_amount = last_serialized_file_number * setting.hyper_params_info.batch_size

    with open(setting.paths_info.metadata_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        _skip_processed_records(overall_processed_unit_amount, reader)
        new_batch = _form_batch_of_units(reader, setting, morph, overall_processed_unit_amount)
        while len(new_batch) == setting.hyper_params_info.batch_size:
            # Save current batch
            last_serialized_file_number = last_serialized_file_number + 1
            _log.info("Formed next batch with number " + last_serialized_file_number.__str__())
            file_path = _serialize_batch(new_batch, setting.paths_info.serialized_units_dir_path,
                                         last_serialized_file_number)
            overall_processed_unit_amount = overall_processed_unit_amount + len(new_batch)
            paths.append(file_path)
            _log.info("Serialized batch " + last_serialized_file_number.__str__() + " to " + file_path)

            # Create new batch
            new_batch = _form_batch_of_units(reader, setting, morph, overall_processed_unit_amount)

        _log.info("Got last batch with size " + len(new_batch).__str__())
    return PreprocessResult(paths=paths)


def _skip_processed_records(processed_unit_amount, reader):
    next(reader, None)  # skip the headers
    if processed_unit_amount > 0:
        for _ in range(processed_unit_amount - 1):
            next(reader)


def _form_batch_of_units(reader, setting: TrainingSetting, morph, overall_processed_unit_amount: int) -> [TrainingUnit]:
    result = []
    processed_unit_amount = 0
    while processed_unit_amount < setting.hyper_params_info.batch_size:
        row = _next_row(reader)
        if len(row) == 0:
            _log.info("No more records in csv file, return result array of " + len(result).__str__() + " size")
            return result

        unit = _form_training_unit(row, setting, morph)
        result.append(unit)
        processed_unit_amount = processed_unit_amount + 1
        _log.info(
            "№ " + (overall_processed_unit_amount + processed_unit_amount).__str__() + "." +
            "Formed training unit from " + row[1].__str__() + "." +
            " Unit " + unit.serialize().__str__()
        )
    return result


def _serialize_batch(batch: [TrainingUnit], serialized_dir_path: str, last_serialized_file_number: int) -> str:
    path = serialized_dir_path + '/serialized_batch_' + last_serialized_file_number.__str__() + '.pkl'
    with open(path, 'ab') as f:
        pickle.dump(batch, f)
        f.close()
    return path


def _next_row(reader) -> list[str]:
    try:
        return next(reader)
    except StopIteration:
        return []


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


def _get_mfcc_db(audio_data: ndarray, sampling_rate: float, setting: TrainingSetting) -> ndarray:
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate, n_mels=setting.num_mels, fmin=125,
                                              fmax=7600)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def _get_spectrogram(audio_data: ndarray, frame_length: int, hop_length: int) -> ndarray:
    return librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length)


def _phonemize_text(text: str, morph: MorphAnalyzer, language: str) -> list[str]:
    words = re.findall(Config.WORDS_REGEX, text)
    all_phonemes = []

    for word in words:
        base_form = morph.parse(word)[0].normal_form
        phonemes = phonemize(base_form, language=language)
        all_phonemes.append(phonemes)

    return all_phonemes


preprocess_audio(ts)
