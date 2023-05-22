from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingHyperParamsInfo:
    batch_size: int
    learning_rate: float
    num_epochs: int
    steps_per_epoch: int
    loss_fun: Any
    validation_split: float
    encoder_layers: int
    decoder_layers: int
    post_kernel_size: int

    def serialize(self):
        return {
            'name': 'TrainingHyperParamInfo',
            'batch_size': self.batch_size,
            'steps_per_epoch': self.steps_per_epoch,
            'loss_fun': self.loss_fun,
            'validation_split': self.validation_split,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'encoder_layers': self.encoder_layers,
            'decoder_layers': self.decoder_layers,
            'post_kernel_size': self.post_kernel_size,
        }
