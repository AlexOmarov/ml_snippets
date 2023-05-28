import numpy as np
from numpy import ndarray
from phonemizer.backend import EspeakBackend
from pymorphy2 import MorphAnalyzer

from business.audio.generation.train.config.dto.training_setting import TrainingSetting
from business.audio.generation.train.dataset.dto.audio_entry import AudioEntry
from business.audio.generation.train.dataset.dto.audio_entry_metadata import AudioEntryMetadata
from business.audio.generation.preprocessing.audio_retrieval import retrieve_via_path
from business.audio.generation.preprocessing.preprocessing import get_phonemes, get_mel_spectrum_db, get_feature_vector
from business.util.ml_logger import logger

_log = logger.get_logger(__name__.replace('__', '\''))


def form_audio_entry(serialized_metadata: list[str],
                     setting: TrainingSetting,
                     morph: MorphAnalyzer,
                     speakers: ndarray,
                     backend: EspeakBackend) -> AudioEntry:
    # Form metadata from serialized row
    speaker_id = _get_speaker_id(serialized_metadata)

    metadata = AudioEntryMetadata(
        audio_path=_get_file_path(setting.paths_info.audio_files_dir_path, serialized_metadata),
        text=_get_text(serialized_metadata),
        sampling_rate=_get_sampling_rate(serialized_metadata),
        duration_seconds=_get_duration(serialized_metadata),
        speaker_id=speaker_id,
    )

    # Load and normalize audio from file
    audio, sr = retrieve_via_path(metadata.audio_path)

    # Get phonemes for text
    phonemes = get_phonemes(metadata.text, "", morph, backend)

    # Get features of audio
    mel_spectrum = get_mel_spectrum_db(audio, metadata.sampling_rate, setting.hyper_params_info.num_mels)

    feature_vector = get_feature_vector(
        audio,
        sr,
        setting.hyper_params_info.frame_length,
        setting.hyper_params_info.hop_length,
        setting.hyper_params_info.num_mels
    )

    return AudioEntry(
        metadata=metadata,
        feature_vector=feature_vector,
        speaker_identification_vector=_get_identification_vector(speaker_id, speakers),
        mel_spectrogram_result=mel_spectrum,
        phonemes=phonemes,
    )


def _get_identification_vector(speaker_id: str, speakers: ndarray) -> ndarray:
    result = np.zeros_like(speakers, dtype=int)
    result[speakers == speaker_id] = 1
    return result


def _get_speaker_id(serialized_metadata: list[str]) -> str:
    return serialized_metadata[2]


def _get_sampling_rate(serialized_metadata: list[str]) -> float:
    return float(serialized_metadata[10])


def _get_duration(serialized_metadata: list[str]) -> float:
    return float(serialized_metadata[8])


def _get_file_path(audio_files_dir_path: str, serialized_metadata: list[str]) -> str:
    return audio_files_dir_path + serialized_metadata[7].removeprefix("audio_files")


def _get_text(serialized_metadata: list[str]) -> str:
    return serialized_metadata[0]
