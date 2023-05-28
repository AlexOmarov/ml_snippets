from dataclasses import dataclass


@dataclass
class AudioEntryMetadata:
    audio_path: str
    text: str
    sampling_rate: float
    duration_seconds: float
    speaker_id: str

    def serialize(self):
        return {
            'name': 'AudioEntryMetadata',
            'audio_path': self.audio_path,
            'sampling_rate': self.sampling_rate,
            'duration_seconds': self.duration_seconds,
            'speaker_id': self.speaker_id,
            'text': self.text
        }
