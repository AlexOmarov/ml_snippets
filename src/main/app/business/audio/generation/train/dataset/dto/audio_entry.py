from dataclasses import dataclass

from numpy import ndarray

from business.audio.generation.train.train import AudioEntryMetadata


@dataclass
class AudioEntry:
    metadata: AudioEntryMetadata
    feature_vector: ndarray
    phonemes: [str]
    speaker_identification_vector: ndarray
    mel_spectrogram_result: ndarray

    def serialize(self):
        return {
            'name': 'AudioEntry',
            'metadata': self.metadata.serialize(),
            'phonemes': self.phonemes,
        }
