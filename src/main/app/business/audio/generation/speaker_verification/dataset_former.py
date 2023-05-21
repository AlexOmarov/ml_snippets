import csv
import os
import pickle

import pymorphy2

from business.audio.generation.config.training_setting import ts
from business.audio.generation.dto.training_setting import TrainingSetting
from business.audio.generation.dto.audio_entry import AudioEntry
from business.audio.generation.speaker_verification.audio_entry_former import form_audio_entry
from business.util.ml_logger import logger
from presentation.api.preprocess_result import PreprocessResult

_log = logger.get_logger(__name__.replace('__', '\''))


def preprocess_audio(setting: TrainingSetting) -> PreprocessResult:
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


def _form_batch_of_units(reader, setting: TrainingSetting, morph, overall_processed_unit_amount: int) -> [AudioEntry]:
    result = []
    processed_unit_amount = 0
    while processed_unit_amount < setting.hyper_params_info.batch_size:
        row = _next_row(reader)
        if len(row) == 0:
            _log.info("No more records in csv file, return result array of " + len(result).__str__() + " size")
            return result

        unit = form_audio_entry(row, setting, morph)
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


preprocess_audio(ts)
