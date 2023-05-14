from dataclasses import dataclass


@dataclass
class PreprocessResult:
    paths: list[str]

    def serialize(self):
        return {
            'paths': self.paths
        }
