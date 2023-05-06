import os

import speech_recognition as sr

from src.main.resource.config import Config

root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))
input_dir = os.path.join(root_dir, Config.VOICE_PATH)


def recognize():
    recognizer = sr.Recognizer()
    with open(os.path.join(root_dir, 'data/words.txt'), 'w') as f:
        for file_name in os.listdir(input_dir):
            with sr.AudioFile(os.path.join(input_dir, file_name)) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language="ru-RU")
                f.write(f'{file_name}|{text}\n')
                print("Ended up for " + file_name + " with text " + text)


recognize()
