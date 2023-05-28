from dataclasses import dataclass


@dataclass
class AudioGenerationResult:
    file_path: str

    def serialize(self):
        return {
            'file_path': self.file_path,
        }
