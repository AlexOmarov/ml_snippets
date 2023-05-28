from dataclasses import dataclass

from business.audio.generation.train.train import TrainingHyperParamsInfo
from business.audio.generation.train.train import TrainingPathsInfo


@dataclass
class TrainingSetting:
    hyper_params_info: TrainingHyperParamsInfo
    paths_info: TrainingPathsInfo
    model_name: str
    words_regex: str
    language: str

    def serialize(self):
        return {
            'name': 'TrainingSetting',
            'hyper_params_info': self.hyper_params_info,
            'phonemize_language': self.language,
            'paths_info': self.paths_info,
            'language': self.language,
            'words_regex': self.words_regex
        }
