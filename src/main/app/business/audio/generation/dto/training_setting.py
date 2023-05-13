from dataclasses import dataclass

from business.audio.generation.dto.training_hyper_params_info import TrainingHyperParamsInfo
from business.audio.generation.dto.training_paths_info import TrainingPathsInfo


@dataclass
class TrainingSetting:
    hyper_params_info: TrainingHyperParamsInfo
    paths_info: TrainingPathsInfo
    model_name: str
    num_mels: int
    vocab_size: int
    frame_length: int
    hop_length: int
    phonemize_language: str

    def serialize(self):
        return {
            'name': 'TrainingSetting',
            'hyper_params_info': self.hyper_params_info,
            'num_mels': self.num_mels,
            'vocab_size': self.vocab_size,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'phonemize_language': self.phonemize_language,
            'paths_info': self.paths_info
        }
