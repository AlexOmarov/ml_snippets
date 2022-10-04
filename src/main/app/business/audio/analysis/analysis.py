"""
Script for extracting features of audio file

This script allows user to extract time/frequency domain features from audio file and save it as a plot.

This tool accepts audio file, frame and hop length as input parameters.

This script requires that `tensorflow` be installed within the Python environment.

This file can also be imported as a module.

Public interface:

    * analyse - gets audio file, parses it as bytes and extracts features into plots.
                Returns paths to created plots
"""

import librosa
import librosa.display
import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from werkzeug.datastructures import FileStorage

from business.util.ml_logger import logger
from presentation.api.audio_analysis_result import AudioAnalysisResult
from src.main.resource.config import Config

_log = logger.get_logger(__name__.replace('__', '\''))
_COLOR_BAR = "%+2.f"


def analyse(storage: FileStorage, frame_length: int, hop_length: int) -> AudioAnalysisResult:
    """
    Performs analyse of the requested audio file.
    Creates time-based, frequency-based graphs and spectrogram

    :param storage: Storage with audio file
    :param frame_length: Length of a single frame
    :param hop_length: Length of a single hop

    :return AudioAnalysisResult Object containing paths to built plots

    :exception NotImplementedError If passed metric isn't supported.

    """
    audio, sr = librosa.load(storage)
    filter_banks = librosa.filters.mel(n_fft=2048, sr=sr, n_mels=10)
    # TODO: Audio data must be of type numpy.ndarray
    mel_spectrogram = librosa.feature.melspectrogram(scale=audio, n_fft=2048, sr=sr, hop_length=512, n_mels=90)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    ae = _get_amplitude_envelope(audio, frame_length, hop_length)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms = _get_rms(audio, frame_length=frame_length, hop_length=hop_length)
    frames = range(0, ae.size)
    t = librosa.frames_to_time(frames, hop_length=hop_length)

    ft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.absolute(ft)
    frequency = np.linspace(0, sr, len(magnitude))

    return AudioAnalysisResult(
        time_features_plot_path=_build_time_features_plot(ae, zcr, rms, audio, sr, storage.filename, t),
        freq_features_plot_path=_build_freq_features_plot(frequency, magnitude, storage.filename),
        spectrogram_plot_path= _build_spectrogram_plot(magnitude, sr, storage.filename, hop_length),
        mel_spectrogram_plot_path= _build_mel_banks_plot(filter_banks, sr, storage.filename),
        mel_banks_plot_path= _build_mel_spectrogram_plot(log_mel_spectrogram, sr, storage.filename),
        frame_length=frame_length,
        hop_length=hop_length
    )


def _build_mel_spectrogram_plot(log_mel_spectrogram, sr, filename: str) -> str:
    path = Config.MODEL_PATH + filename + '_mel_spectrogram.png'
    plt.figure(figsize=(25, 10))

    plt.subplot(3, 1, 1)

    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format=_COLOR_BAR)
    plt.title(filename)
    plt.savefig(path)

    return path


def _build_mel_banks_plot(filter_banks, sr, filename: str) -> str:
    path = Config.MODEL_PATH + filename + '_mel_banks.png'
    plt.figure(figsize=(25, 10))

    plt.subplot(3, 1, 1)

    librosa.display.specshow(filter_banks, sr=sr, x_axis="linear")
    plt.colorbar(format=_COLOR_BAR)
    plt.title(filename)
    plt.savefig(path)

    return path


def _build_spectrogram_plot(magnitude, sr, filename: str, hop_length: int) -> str:
    path = Config.MODEL_PATH + filename + '_spectrogram.png'
    plt.figure(figsize=(15, 17))

    plt.subplot(3, 1, 1)

    librosa.display.specshow(
        librosa.power_to_db(magnitude ** 2), sr=sr, hop_length=hop_length, x_axis="time", y_axis="log"
    )
    plt.colorbar(format=_COLOR_BAR)
    plt.title(filename)
    plt.savefig(path)

    return path


def _build_freq_features_plot(frequency, magnitude, filename: str) -> str:
    path = Config.MODEL_PATH + filename + '_freq_features.png'
    plt.figure(figsize=(15, 17))

    plt.subplot(3, 1, 1)

    plt.plot(frequency[:5000], magnitude[:5000])
    plt.xlabel("Freq (Hz)")
    plt.ylabel("Magnitude")
    plt.title(filename)
    plt.savefig(path)
    return path


def _build_time_features_plot(ae, zcr, rms, audio, sr, filename: str, t) -> str:
    path = Config.MODEL_PATH + filename + '_features.png'
    plt.figure(figsize=(15, 17))

    plt.subplot(3, 1, 1)

    red_patch = mpl_patches.Patch(color='red', label='Amplitude envelope')
    blue_patch = mpl_patches.Patch(color='blue', label='Waveform')
    green_patch = mpl_patches.Patch(color='green', label='RMS energy')
    yellow_patch = mpl_patches.Patch(color='yellow', label='Zero crossing rate')
    plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch], fontsize=10, shadow=True, framealpha=0.5,
               edgecolor='b',
               title='Legend')

    librosa.display.waveshow(audio, sr=sr, alpha=0.5)
    plt.plot(t, ae, color="r")
    plt.plot(t, rms, color="g")
    plt.plot(t, zcr, color="y")
    plt.title(filename)
    plt.ylim((-1, 1))
    plt.savefig(path)
    return path


def _get_amplitude_envelope(signal, frame_length: int, hop_length: int) -> ndarray:
    result = []
    for i in range(0, len(signal), hop_length):
        current_ae = max(signal[i:i + frame_length])
        result.append(current_ae)
    return np.array(result)


def _get_rms(signal, frame_length, hop_length) -> ndarray:
    result = []
    for i in range(0, len(signal), hop_length):
        current_rms = np.sqrt(np.sum(signal[i:i + frame_length] ** 2) / frame_length)
        result.append(current_rms)
    return np.array(result)
