import os
import shutil

from pydub import AudioSegment

input_dir = 'C:\\Users\\shtil\\Desktop\\ogg'

output_dir = 'C:\\Users\\shtil\\Desktop\\wavs'

input_ext = '.ogg'
output_ext = '.wav'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    if os.path.isfile(input_path):
        output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + output_ext)
        # shutil.copy(input_path, output_path)
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
