from dataclasses import dataclass


@dataclass
class ClassificationResult:
    imageName: str
    label: str

    def serialize(self):
        return {
            'imageName': self.imageName,
            'label': self.label
        }
