import pymorphy2
from flask import g
from phonemizer.backend import EspeakBackend


def get_morph():
    if 'morph' not in g:
        g.morph = pymorphy2.MorphAnalyzer()
    return g.morph


def get_espeak_backend(language: str):
    if 'backend_' + language not in g:
        g.backend = EspeakBackend(language, preserve_punctuation=True)
    return g.backend
