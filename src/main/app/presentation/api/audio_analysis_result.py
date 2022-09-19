from dataclasses import dataclass


@dataclass
class AudioAnalysisResult:
    visuals_path: str
    frame_length: int
    hop_length: int

    def serialize(self):
        return {
            'metric': self.metric,
            'visuals_path': self.visuals_path
        }
