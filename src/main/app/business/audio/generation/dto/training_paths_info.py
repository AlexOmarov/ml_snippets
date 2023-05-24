from dataclasses import dataclass


@dataclass
class TrainingPathsInfo:
    metadata_file_path: str
    speaker_file_path: str
    serialized_units_dir_path: str
    serialized_test_units_dir_path: str
    phonemes_file_path: str
    audio_files_dir_path: str
    checkpoint_path_template: str
    model_dir_path: str

    def serialize(self):
        return {
            'name': 'TrainingPathsInfo',
            'metadata_file_path': self.metadata_file_path,
            'serialized_units_dir_path': self.serialized_units_dir_path,
            'serialized_test_units_dir_path': self.serialized_test_units_dir_path,
            'speaker_file_path': self.speaker_file_path,
            'audio_files_dir_path': self.audio_files_dir_path,
            'checkpoint_path_template': self.checkpoint_path_template,
            'phonemes_file_path': self.phonemes_file_path,
            'model_dir_path': self.model_dir_path,
        }
