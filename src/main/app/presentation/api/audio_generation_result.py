from dataclasses import dataclass


@dataclass
class AudioGenerationResult:
    data: str

    def serialize(self):
        return {
            'data': self.data,
        }
