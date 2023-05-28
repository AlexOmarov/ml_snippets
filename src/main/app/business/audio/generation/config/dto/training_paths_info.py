from dataclasses import dataclass


@dataclass
class TrainingPathsInfo:
    metadata_file_path: str

    def serialize(self):
        return {
            'name': 'TrainingPathsInfo',
            'metadata_file_path': self.metadata_file_path,
        }
