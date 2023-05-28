import os
import re

import pymorphy2
import speech_recognition as sr
from phonemizer import phonemize


def write_text_from_audio(destination_path: str,
                          audio_files_dir_path: str,
                          recognize_language: str,
                          phonemize_language: str,
                          phonemize_backend: str,
                          encoding: str):
    recognizer = sr.Recognizer()
    morph = pymorphy2.MorphAnalyzer()

    with open(destination_path, 'w', encoding=encoding) as words_file:
        for file_name in os.listdir(audio_files_dir_path):
            text = _recognize_text(os.path.join(audio_files_dir_path, file_name), recognizer, recognize_language)
            phonemes = _phonemize_text(text, morph, phonemize_language, phonemize_backend)
            words_file.write(f'{file_name}|{text}|{phonemes}\n')


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
