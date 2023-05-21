import re

import librosa
import numpy as np
from numpy import ndarray
from phonemizer.backend import EspeakBackend
from pymorphy2 import MorphAnalyzer

from business.audio.generation.dto.audio_entry import AudioEntry
from business.audio.generation.dto.audio_entry_metadata import AudioEntryMetadata
from business.audio.generation.dto.training_setting import TrainingSetting
from business.util.ml_logger import logger
from src.main.resource.config import Config

_log = logger.get_logger(__name__.replace('__', '\''))


def form_audio_entry(serialized_metadata: list[str], setting: TrainingSetting, morph: MorphAnalyzer,
                     speakers: ndarray,
                     backend: EspeakBackend) -> AudioEntry:
    # Form metadata from serialized row
    speaker_id = _get_speaker_id(serialized_metadata)
    metadata = AudioEntryMetadata(
        audio_path=_get_file_path(setting, serialized_metadata),
        text=_get_text(serialized_metadata),
        sampling_rate=_get_sampling_rate(serialized_metadata),
        duration_seconds=_get_duration(serialized_metadata),
        speaker_id=speaker_id,
    )

    # Load and normalize audio from file
    audio = _get_audio(metadata.audio_path, metadata.sampling_rate, metadata.duration_seconds)

    # Get phonemes for text
    phonemes = _phonemize_text(metadata.text, morph, backend)

    # Get features of audio
    mel_spectrogram = _get_mel_spectrogram(audio, metadata.sampling_rate, setting)
    feature_vector = _get_feature_vector(audio, metadata, setting, mel_spectrogram)

    return AudioEntry(
        metadata=metadata,
        feature_vector=feature_vector,
        speaker_identification_vector=_get_identification_vector(speaker_id, speakers),
        mel_spectrogram_result=mel_spectrogram,
        phonemes=phonemes,
    )


def _get_identification_vector(speaker_id: str, speakers: ndarray) -> ndarray:
    result = np.zeros_like(speakers, dtype=int)
    result[speakers == speaker_id] = 1
    return result


def _get_feature_vector(audio, metadata, setting, mel_spectrogram) -> ndarray:
    spectrogram = _get_spectrogram(audio, setting.frame_length, setting.hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(S=spectrogram, sr=metadata.sampling_rate)
    chromagram = librosa.feature.chroma_cqt(y=audio, sr=metadata.sampling_rate)
    tonnetz = librosa.feature.tonnetz(chroma=chromagram)
    mfccs = librosa.feature.mfcc(y=audio, sr=metadata.sampling_rate)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    result = []
    for entry in spectral_contrast:
        result.append(np.mean(entry))
    for entry in mel_spectrogram:
        result.append(np.mean(entry))
    for entry in tonnetz:
        result.append(np.mean(entry))
    for entry in chromagram:
        result.append(np.mean(entry))
    for entry in mfccs:
        result.append(np.mean(entry))
    for entry in delta_mfccs:
        result.append(np.mean(entry))
    for entry in delta2_mfccs:
        result.append(np.mean(entry))
    return np.array(result)


def _get_mel_spectrogram(audio_data: ndarray, sampling_rate: float, setting: TrainingSetting) -> ndarray:
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate, n_mels=setting.num_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def _get_speaker_id(serialized_metadata: list[str]) -> str:
    return serialized_metadata[2]


def _get_file_path(setting: TrainingSetting, serialized_metadata: list[str]) -> str:
    return setting.paths_info.audio_files_dir_path + serialized_metadata[7].removeprefix("audio_files")


def _get_sampling_rate(serialized_metadata: list[str]) -> float:
    return float(serialized_metadata[9])


def _get_duration(serialized_metadata: list[str]) -> float:
    return float(serialized_metadata[10])


def _get_text(serialized_metadata: list[str]) -> str:
    return serialized_metadata[0]


def _get_audio(path: str, sampling_rate: float, duration: float) -> ndarray:
    audio_data, sr = librosa.load(path, sr=sampling_rate, duration=duration, mono=True)

    # Normalize the audio
    audio_data /= max(abs(audio_data))

    # Trim leading and trailing silence
    audio_data, _ = librosa.effects.trim(audio_data)

    # Resample the audio if necessary
    if sr != sampling_rate:
        audio_data = librosa.resample(audio_data, sr, sampling_rate)
    return audio_data


def _get_spectrogram(audio_data: ndarray, frame_length: int, hop_length: int) -> ndarray:
    return np.abs(librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length))


def _phonemize_text(text: str, morph: MorphAnalyzer, backend: EspeakBackend) -> list[str]:
    words = re.findall(Config.WORDS_REGEX, text)
    base_form_words = []

    for word in words:
        base_form = morph.parse(word)[0].normal_form
        base_form_words.append(base_form)
    phonemes = backend.phonemize(base_form_words)
    return phonemes
