from dataclasses import dataclass

from business.audio.generation.config.dto.training_hyper_params_info import TrainingHyperParamsInfo
from business.audio.generation.config.dto.training_paths_info import TrainingPathsInfo


@dataclass
class TrainingSetting:
    hyper_params_info: TrainingHyperParamsInfo
    paths_info: TrainingPathsInfo
    model_name: str
    language: str

    def serialize(self):
        return {
            'name': 'TrainingSetting',
            'hyper_params_info': self.hyper_params_info,
            'phonemize_language': self.language,
            'paths_info': self.paths_info
        }
