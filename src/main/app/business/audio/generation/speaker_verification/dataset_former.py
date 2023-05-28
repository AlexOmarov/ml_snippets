import csv
import os
import pickle

import numpy as np
import pymorphy2
from numpy import ndarray
from phonemizer.backend import EspeakBackend

from business.audio.generation.config.dto.training_setting import TrainingSetting
from business.audio.generation.dataset.dto.audio_entry import AudioEntry
from business.audio.generation.speaker_verification.audio_entry_former import form_audio_entry
from business.util.ml_logger import logger

_log = logger.get_logger(__name__.replace('__', '\''))


def preprocess_audio(setting: TrainingSetting):
    paths = []
    morph = pymorphy2.MorphAnalyzer()
    backend = EspeakBackend(setting.phonemize_language, preserve_punctuation=True)
    # Get overall_processed_unit_amount (batch size should be same between calls)
    serialized_files = os.listdir(setting.paths_info.serialized_units_dir_path)
    last_serialized_file_number = len(serialized_files)
    overall_processed_unit_amount = last_serialized_file_number * setting.hyper_params_info.batch_size
    speakers = _get_speakers(setting.paths_info.speaker_file_path)
    with open(setting.paths_info.metadata_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        _skip_processed_records(overall_processed_unit_amount, reader)
        new_batch = _form_batch_of_units(reader, setting, morph, overall_processed_unit_amount, speakers, backend)
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
            new_batch = _form_batch_of_units(reader, setting, morph, overall_processed_unit_amount, speakers, backend)

        _log.info("Got last batch with size " + len(new_batch).__str__())


def _get_speakers(path: str) -> ndarray:
    speakers = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        row = _next_row(reader)
        while row:
            speakers.append(row[0])
            row = _next_row(reader)
    return np.array(speakers)


def _get_speakers(metadata_file_path: str, speaker_file_path: str):
    result = read(metadata_file_path, True)
    distinct_values = list(dict.fromkeys(result))
    write(distinct_values, speaker_file_path)


def _skip_processed_records(processed_unit_amount, reader):
    next(reader, None)  # skip the headers
    if processed_unit_amount > 0:
        for _ in range(processed_unit_amount - 1):
            next(reader)


def _form_batch_of_units(reader, setting: TrainingSetting, morph, overall_processed_unit_amount: int,
                         speakers: ndarray,
                         backend: EspeakBackend) -> [AudioEntry]:
    result = []
    processed_unit_amount = 0
    while processed_unit_amount < setting.hyper_params_info.batch_size:
        row = _next_row(reader)
        if len(row) == 0:
            _log.info("No more records in csv file, return result array of " + len(result).__str__() + " size")
            return result

        unit = form_audio_entry(row, setting, morph, speakers, backend)
        result.append(unit)
        processed_unit_amount = processed_unit_amount + 1
        _log.info(
            "№ " + (overall_processed_unit_amount + processed_unit_amount).__str__() + "." +
            "Formed training unit from " + row[1].__str__() + "." +
            " Unit " + unit.serialize().__str__()
        )
    return result


def _serialize_batch(batch: [AudioEntry], serialized_dir_path: str, last_serialized_file_number: int) -> str:
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
