import librosa
from numpy import ndarray
from werkzeug.datastructures import FileStorage


def retrieve_via_path(path: str, required_sample_rate: int) -> tuple[ndarray, int]:
    audio_data, sr = librosa.load(path)
    return _normalize(audio_data, sr, required_sample_rate), required_sample_rate


def retrieve_via_storage(storage: FileStorage, required_sample_rate: int) -> tuple[ndarray, int]:
    audio_data, sr = librosa.load(storage)
    return _normalize(audio_data, sr, required_sample_rate), sr


def _normalize(audio_data, sr: int, required_sample_rate: int) -> ndarray:
    # Normalize the audio
    audio_data /= max(abs(audio_data))
    # Trim leading and trailing silence
    audio_data, _ = librosa.effects.trim(audio_data)

    # Resample the audio if necessary
    if sr != required_sample_rate:
        audio_data = librosa.resample(audio_data, sr, required_sample_rate)

    return audio_data
