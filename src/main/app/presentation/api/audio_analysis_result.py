from dataclasses import dataclass


@dataclass
class AudioAnalysisResult:
    time_features_plot_path: str
    freq_features_plot_path: str
    spectrogram_plot_path: str
    frame_length: int
    hop_length: int

    def serialize(self):
        return {
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'spectrogram_plot_path': self.spectrogram_plot_path,
            'time_features_plot_path': self.time_features_plot_path,
            'freq_features_plot_path': self.freq_features_plot_path
        }
