from dataclasses import dataclass

from numpy import ndarray

from business.audio.generation.dto.audio_entry_metadata import AudioEntryMetadata


@dataclass
class AudioEntry:
    metadata: AudioEntryMetadata
    feature_vector: ndarray
    speaker_identification_vector: ndarray
    mel_spectrogram_result: ndarray
    phonemes: [str]

    def serialize(self):
        return {
            'name': 'AudioEntry',
            'metadata': self.metadata.serialize(),
            'phonemes': self.phonemes,
        }
