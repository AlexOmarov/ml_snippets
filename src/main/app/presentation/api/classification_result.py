from dataclasses import dataclass


@dataclass
class ClassificationResult:
    image_name: str
    label: str

    def serialize(self):
        return {
            'imageName': self.image_name,
            'label': self.label
        }
