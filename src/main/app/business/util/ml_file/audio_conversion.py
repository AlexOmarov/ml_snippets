import os
import shutil

from pydub import AudioSegment

from src.main.resource.config import Config


def convert(make_conversion: bool,
            input_dir: str = Config.AUDIO_CONVERSION_INPUT_DIR_PATH,
            output_ext: str = Config.AUDIO_CONVERSION_OUTPUT_EXT,
            output_dir: str = Config.AUDIO_FILES_DIR_PATH):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        if os.path.isfile(input_path):
            output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + "." + output_ext)
            if make_conversion:
                audio = AudioSegment.from_file(input_path)
                audio.export(output_path, format=output_ext)
            else:
                shutil.copy(input_path, output_path)
