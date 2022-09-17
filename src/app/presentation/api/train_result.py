from dataclasses import dataclass


@dataclass
class TrainResult:
    metric: str
    data: list

    def serialize(self):
        return {
            'metric': self.metric,
            'data': self.data
        }
