import csv
import random

from src.main.resource.config import Config


def _next_row(reader) -> list[str]:
    try:
        return next(reader)
    except StopIteration:
        return []


result = []
with open(Config.METADATA_FILE_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    row = _next_row(reader)
    while row:
        result.append(row)
        row = _next_row(reader)
random.shuffle(result)

with open(Config.METADATA_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(result)
