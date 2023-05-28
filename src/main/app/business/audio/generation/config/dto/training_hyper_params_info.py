from dataclasses import dataclass


@dataclass
class TrainingHyperParamsInfo:
    batch_size: int
    num_epochs: int
    steps_per_epoch: int

    def serialize(self):
        return {
            'name': 'TrainingHyperParamInfo',
            'batch_size': self.batch_size,
            'steps_per_epoch': self.steps_per_epoch,
            'num_epochs': self.num_epochs,
        }
