from dataclasses import dataclass

from numpy import ndarray

from business.audio.generation.dto.training_hyper_params_info import TrainingHyperParamsInfo
from business.audio.generation.dto.training_paths_info import TrainingPathsInfo


@dataclass
class TrainingUnit:
    audio_path: str
    text: str
    phonemes: [str]
    sampling_rate: float
    duration_seconds: float
    mfcc_db: ndarray
    spectrogram: ndarray

    def serialize(self):
        return {
            'name': 'TrainingUnit',
            'audio_path': self.audio_path,
            'phonemes': self.phonemes,
            'sampling_rate': self.sampling_rate,
            'duration_seconds': self.duration_seconds,
            'text': self.text
        }
