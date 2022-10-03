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


def analyse(storage: FileStorage, frame_length: int, hop_length: int) -> AudioAnalysisResult:
    """
    Performs analyse of the requested audio file

    :param storage: Storage with audio file
    :param frame_length: Length of a single frame
    :param hop_length: Length of a single hop

    :return AudioAnalysisResult

    :exception NotImplementedError If passed metric isn't supported.

    """
    audio, sr = librosa.load(storage)

    time_feature_plot_path: str = _build_time_features_plot(audio, sr, storage.filename, frame_length, hop_length)
    freq_feature_plot_path: str = _build_freq_features_plot(audio, sr, storage.filename, frame_length, hop_length)

    return AudioAnalysisResult(time_feature_plot_path, freq_feature_plot_path, frame_length, hop_length)


def _build_freq_features_plot(audio, sr, filename: str, frame_length: int, hop_length: int) -> str:
    ft = _get_fft(audio, frame_length, hop_length)
    magnitude = np.absolute(ft)
    frequency = np.linspace(0, sr, len(magnitude))
    _build_spectrogram_plot(magnitude, filename, sr, hop_length)
    return _build_freq_plot(magnitude, frequency, filename)


def _build_time_features_plot(audio, sr, filename: str, frame_length: int, hop_length: int) -> str:
    ae = _get_amplitude_envelope(audio, frame_length, hop_length)
    zcr = _get_zcr(audio, frame_length, hop_length)
    rms = _get_rms(audio, frame_length=frame_length, hop_length=hop_length)

    return _build_plot(audio, ae, zcr, rms, filename, sr, hop_length)


def _get_amplitude_envelope(signal, frame_length: int, hop_length: int) -> ndarray:
    result = []
    for i in range(0, len(signal), hop_length):
        current_ae = max(signal[i:i + frame_length])
        result.append(current_ae)
    return np.array(result)


def _get_zcr(audio, frame_length, hop_length):
    return librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]


def _get_fft(audio, frame_length: int, hop_length: int):
    return librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)


def _get_rms(signal, frame_length, hop_length) -> ndarray:
    result = []
    for i in range(0, len(signal), hop_length):
        current_rms = np.sqrt(np.sum(signal[i:i + frame_length] ** 2) / frame_length)
        result.append(current_rms)
    return np.array(result)


def _build_freq_plot(magnitude, frequency, filename: str) -> str:
    path = Config.MODEL_PATH + filename + '_freq_features.png'
    plt.figure(figsize=(15, 17))

    plt.subplot(3, 1, 1)

    plt.plot(frequency[:5000], magnitude[:5000])
    plt.xlabel("Freq (Hz)")
    plt.ylabel("Magnitude")
    plt.title(filename)
    plt.savefig(path)
    return path


def _build_spectrogram_plot(magnitude, filename: str, sr, hop_length: int) -> str:
    path = Config.MODEL_PATH + filename + '_spectrogram.png'
    plt.figure(figsize=(15, 17))

    plt.subplot(3, 1, 1)

    librosa.display.specshow(
        librosa.power_to_db(magnitude ** 2), sr=sr, hop_length=hop_length, x_axis="time", y_axis="log"
    )
    plt.colorbar(format="%+2.f")
    plt.title(filename)
    plt.savefig(path)

    return path


def _build_plot(signal, ae, zcr, rms, filename: str, sr: int, hop_length: int) -> str:
    path = Config.MODEL_PATH + filename + '_features.png'
    frames = range(0, ae.size)
    t = librosa.frames_to_time(frames, hop_length=hop_length)
    plt.figure(figsize=(15, 17))

    plt.subplot(3, 1, 1)

    red_patch = mpl_patches.Patch(color='red', label='Amplitude envelope')
    blue_patch = mpl_patches.Patch(color='blue', label='Waveform')
    green_patch = mpl_patches.Patch(color='green', label='RMS energy')
    yellow_patch = mpl_patches.Patch(color='yellow', label='Zero crossing rate')
    plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch], fontsize=10, shadow=True, framealpha=0.5,
               edgecolor='b',
               title='Legend')

    librosa.display.waveshow(signal, sr=sr, alpha=0.5)
    plt.plot(t, ae, color="r")
    plt.plot(t, rms, color="g")
    plt.plot(t, zcr, color="y")
    plt.title(filename)
    plt.ylim((-1, 1))
    plt.savefig(path)
    return path
