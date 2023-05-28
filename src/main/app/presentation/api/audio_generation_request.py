from dataclasses import dataclass


@dataclass
class AudioGenerationRequest:
    text: str

    def serialize(self):
        return {
            'text': self.text
        }
