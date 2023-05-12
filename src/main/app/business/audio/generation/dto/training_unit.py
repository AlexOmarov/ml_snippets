from dataclasses import dataclass

from business.audio.generation.dto.training_hyper_params_info import TrainingHyperParamsInfo
from business.audio.generation.dto.training_paths_info import TrainingPathsInfo


@dataclass
class TrainingUnit:
    audio_path: str
    text: str
    phonemes: [str]
    sample_rate: int
    duration_milliseconds: int
    audio: int
    mfcc_db: int
    spectrogram: int

    def serialize(self):
        return {
            'name': 'TrainingUnit',
            'audio_path': self.audio_path,
            'phonemes': self.phonemes,
            'sample_rate': self.sample_rate,
            'duration_milliseconds': self.duration_milliseconds,
            'text': self.text
        }
