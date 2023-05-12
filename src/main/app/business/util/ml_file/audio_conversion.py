import os
import shutil

from pydub import AudioSegment


def convert(input_dir: str, output_ext: str, output_dir_path: str):
    _make_output_dir(output_dir_path)

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        if os.path.isfile(input_path):
            output_path = os.path.join(output_dir_path, os.path.splitext(file_name)[0] + "." + output_ext)
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format=output_ext)


def copy(input_dir: str, output_dir_path: str):
    _make_output_dir(output_dir_path)

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        path_components = os.path.splitext(file_name)
        if os.path.isfile(input_path):
            output_path = os.path.join(output_dir_path, path_components[0] + "." + path_components[1])
            shutil.copy(input_path, output_path)


def _make_output_dir(output_dir_path: str):
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
