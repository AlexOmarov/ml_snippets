import librosa
from numpy import ndarray
from werkzeug.datastructures import FileStorage


def retrieve_via_path(path: str) -> tuple[ndarray, int]:
    audio_data, sr = librosa.load(path)
    return _normalize(audio_data), sr


def retrieve_via_storage(storage: FileStorage) -> tuple[ndarray, int]:
    audio_data, sr = librosa.load(storage)
    return _normalize(audio_data), sr


def _normalize(audio_data) -> ndarray:
    # Normalize the audio
    audio_data /= max(abs(audio_data))
    # Trim leading and trailing silence
    audio_data, _ = librosa.effects.trim(audio_data)

    return audio_data
