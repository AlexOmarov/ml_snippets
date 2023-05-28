import csv
import random


def next_row(reader) -> list[str]:
    try:
        return next(reader)
    except StopIteration:
        return []


def shuffle(file_path: str, skip_headers: bool):
    result = read(file_path, skip_headers)
    random.shuffle(result)
    write(result, file_path)


def write(result: list[list[str]], file_path: str):
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(result)


def read(file_path: str, skip_headers: bool) -> list[list[str]]:
    result = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)

        if skip_headers:
            next(reader, None)

        row = next_row(reader)

        while row:
            result.append(row)
            row = next_row(reader)

    return result


def skip_processed_records(processed, reader, skip_headers: bool):
    if skip_headers:
        next(reader, None)
    if processed > 0:
        for _ in range(processed - 1):
            next(reader)
