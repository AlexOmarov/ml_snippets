from dataclasses import dataclass

from numpy import ndarray

from business.audio.generation.dto.training_unit_metadata import TrainingUnitMetadata


@dataclass
class TrainingUnit:
    metadata: TrainingUnitMetadata
    feature_vector: ndarray
    speaker_identification_result: ndarray
    speech_generation_result: ndarray
    phonemes: [str]

    def serialize(self):
        return {
            'name': 'TrainingUnit',
            'metadata': self.metadata,
            'phonemes': self.phonemes,
        }
