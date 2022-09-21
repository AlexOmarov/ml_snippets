from dataclasses import dataclass


@dataclass
class AudioAnalysisResult:
    visuals_path: str
    frame_length: int
    hop_length: int

    def serialize(self):
        return {
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'visuals_path': self.visuals_path
        }
