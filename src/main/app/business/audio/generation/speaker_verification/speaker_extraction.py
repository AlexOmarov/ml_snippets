import csv

from business.audio.generation.config.training_setting import ts
from business.audio.generation.dto.training_setting import TrainingSetting
from business.util.ml_logger import logger
from src.main.resource.config import Config

_log = logger.get_logger(__name__.replace('__', '\''))


def get_speakers(setting: TrainingSetting):
    result = []
    with open(setting.paths_info.metadata_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        num = 1
        row = _next_row(reader)
        while row:
            result.append(row[2])
            row = _next_row(reader)
            _log.info("№ " + num.__str__() + " " + row.__str__())
            num += 1
    distinct_values = list(dict.fromkeys(result))
    with open(Config.SPEAKER_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write each string as a separate row
        for string in distinct_values:
            writer.writerow([string])
        f.close()


def _next_row(reader) -> list[str]:
    try:
        return next(reader)
    except StopIteration:
        return []


get_speakers(ts)
