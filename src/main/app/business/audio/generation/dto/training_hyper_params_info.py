from dataclasses import dataclass


@dataclass
class TrainingHyperParamsInfo:
    batch_size: int
    learning_rate: float
    num_epochs: int
    loss_fun: int
    encoder_layers = int
    decoder_layers = int
    post_kernel_size = int

    def serialize(self):
        return {
            'name': 'TrainingHyperParamInfo',
            'batch_size': self.batch_size,
            'loss_fun': self.loss_fun,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'encoder_layers': self.encoder_layers,
            'decoder_layers': self.decoder_layers,
            'post_kernel_size': self.post_kernel_size,
        }
