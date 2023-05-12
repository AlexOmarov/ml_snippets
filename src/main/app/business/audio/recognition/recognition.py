import os
import re

import pymorphy2
import speech_recognition as sr
from phonemizer import phonemize

from src.main.resource.config import Config


def write_audio_text_info(words_file_path: str = Config.METADATA_FILE_PATH,
                          audio_files_dir_path: str = Config.AUDIO_FILES_DIR_PATH,
                          recognize_language: str = Config.RECOGNIZE_LANGUAGE,
                          phonemize_language: str = Config.PHONEMIZE_LANGUAGE,
                          phonemize_backend: str = Config.PHONEMIZE_BACKEND,
                          encoding: str = Config.WORDS_FILE_ENCODING):
    recognizer = sr.Recognizer()
    morph = pymorphy2.MorphAnalyzer()

    with open(words_file_path, 'w', encoding=encoding) as words_file:
        for file_name in os.listdir(audio_files_dir_path):
            text = _recognize_text(os.path.join(audio_files_dir_path, file_name), recognizer, recognize_language)
            phonemes = _phonemize_text(text, morph, phonemize_language, phonemize_backend)
            words_file.write(f'{file_name}|{text}|{phonemes}\n')
            print("Ended up for " + file_name)


def _recognize_text(file_path, recognizer, language) -> str:
    with sr.AudioFile(file_path) as source:
        text = recognizer.recognize_google(recognizer.record(source), language=language)
    return text


def _phonemize_text(text, morph, language, backend) -> str:
    words = re.findall(r'\b\w+\b', text)
    all_phonemes = []

    for word in words:
        base_form = morph.parse(word)[0].normal_form
        phonemes = phonemize(base_form, language=language, backend=backend)
        all_phonemes.append(phonemes)

    return ''.join(all_phonemes)
