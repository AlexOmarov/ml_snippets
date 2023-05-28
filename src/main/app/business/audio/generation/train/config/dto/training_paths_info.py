from dataclasses import dataclass


@dataclass
class TrainingPathsInfo:
    metadata_file_path: str
    models_dir_path: str
    audio_files_dir_path: str
    serialized_units_dir_path: str

    def serialize(self):
        return {
            'name': 'TrainingPathsInfo',
            'metadata_file_path': self.metadata_file_path,
            'audio_files_dir_path': self.audio_files_dir_path,
            'serialized_units_dir_path': self.serialized_units_dir_path,
            'models_dir_path': self.models_dir_path,
        }
