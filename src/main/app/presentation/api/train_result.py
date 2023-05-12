from dataclasses import dataclass


@dataclass
class TrainResult:
    metric: str
    path: str
    tflite_path: str
    data: list

    def serialize(self):
        return {
            'metric': self.metric,
            'path': self.path,
            'tflite_path': self.tflite_path,
            'data': self.data
        }
