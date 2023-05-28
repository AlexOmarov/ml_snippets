from dataclasses import dataclass


@dataclass
class TrainingHyperParamsInfo:
    batch_size: int
    num_epochs: int
    steps_per_epoch: int
    num_mels: int
    frame_length: int
    hop_length: int

    def serialize(self):
        return {
            'name': 'TrainingHyperParamInfo',
            'batch_size': self.batch_size,
            'steps_per_epoch': self.steps_per_epoch,
            'num_epochs': self.num_epochs,
            'num_mels': self.num_mels,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
        }
