from dataclasses import dataclass


@dataclass
class TrainingDataset:
    training_data: str
    training_responses: str
    test_data: str
    test_responses: str

    def serialize(self):
        return {
            'name': 'TrainingDataset',
            'training_data': self.training_data,
            'training_responses': self.training_responses,
            'test_data': self.test_data,
            'test_responses': self.test_responses,
        }
