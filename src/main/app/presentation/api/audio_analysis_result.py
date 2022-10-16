from dataclasses import dataclass


@dataclass
class AudioAnalysisResult:
    time_features_plot_path: str
    freq_features_plot_path: str
    spectrogram_plot_path: str
    mel_spectrogram_plot_path: str
    mfcc_plot_path: str
    delta_mfcc_plot_path: str
    delta2_mfcc_plot_path: str
    mel_banks_plot_path: str
    frame_length: int
    hop_length: int

    def serialize(self):
        return {
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'spectrogram_plot_path': self.spectrogram_plot_path,
            'mel_banks_plot_path': self.mel_banks_plot_path,
            'mel_spectrogram_plot_path': self.mel_spectrogram_plot_path,
            'mfcc_plot_path': self.mfcc_plot_path,
            'delta_mfcc_plot_path': self.delta_mfcc_plot_path,
            'delta2_mfcc_plot_path': self.delta2_mfcc_plot_path,
            'time_features_plot_path': self.time_features_plot_path,
            'freq_features_plot_path': self.freq_features_plot_path
        }
