import pymorphy2
from flask import g
from keras.saving.save import load_model
from phonemizer.backend import EspeakBackend


def get_morph():
    if 'morph' not in g:
        g.morph = pymorphy2.MorphAnalyzer()
    return g.morph


def get_espeak_backend(language: str):
    if 'backend_' + language not in g:
        g.backend = EspeakBackend(language, preserve_punctuation=True)
    return g.backend


def get_audio_model(path: str):
    if 'audio_model' not in g:
        g.audio_model = load_model(path)
    return g.audio_model
