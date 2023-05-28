import re

import librosa
import numpy as np
from numpy import ndarray
from phonemizer.backend import EspeakBackend
from pymorphy2 import MorphAnalyzer


def get_phonemes(text: str, words_regex: str, morph: MorphAnalyzer, backend: EspeakBackend) -> list[str]:
    words = re.findall(words_regex, text)
    base_form_words = []

    for word in words:
        base_form = morph.parse(word)[0].normal_form
        base_form_words.append(base_form)
    phonemes = backend.phonemize(base_form_words)
    return phonemes


def get_feature_vector(audio, sr: float, frame_length: int, hop_length: int, num_mels: int) -> ndarray:
    mel_spectrum_db = get_mel_spectrum_db(audio, sr, num_mels)
    spectrum = _get_spectrum(audio, frame_length, hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(S=spectrum, sr=sr)
    chromagram = librosa.feature.chroma_cqt(y=audio, sr=sr)
    tonnetz = librosa.feature.tonnetz(chroma=chromagram)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    result = []
    _append(result, spectral_contrast)
    _append(result, mel_spectrum_db)
    _append(result, tonnetz)
    _append(result, chromagram)
    _append(result, mfccs)
    _append(result, delta_mfccs)
    _append(result, delta2_mfccs)
    return np.array(result)


def get_mel_spectrum_db(audio: ndarray, sampling_rate: float, num_mels: int) -> ndarray:
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=num_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def _get_spectrum(audio: ndarray, frame_length: int, hop_length: int) -> ndarray:
    return np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))


def _append(result: list, array: ndarray):
    for entry in array:
        result.append(np.mean(entry))
