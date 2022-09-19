#  Lib imports
import librosa
import librosa.display
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from werkzeug.datastructures import FileStorage

#  App imports
from business.util.ml_logger import logger
from presentation.api.audio_analysis_result import AudioAnalysisResult
from src.resource.config import Config

_log = logger.get_logger(__name__.replace('__', '\''))


def analyse(storage: FileStorage) -> AudioAnalysisResult:
    audio: str = _process_single_image(storage)
    return AudioAnalysisResult("metric", [audio])


def _process_single_image(storage: FileStorage) -> str:
    audio, sr = librosa.load(storage)
    _get_amplitude_envelope(audio, sr, storage.filename)
    return ""


def _get_amplitude_envelope(audio, sr, filename: str):
    frame_length = 4096
    hop_length = 2048
    ae = _amplitude_envelope(audio, frame_length, hop_length)
    # rms = librosa.feature.rms(audio, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms = _rms(audio, frame_length=frame_length, hop_length=hop_length)
    # TODO: how to get t? What is this time?
    frames = range(0, ae.size)
    t = librosa.frames_to_time(frames, hop_length=hop_length)
    plt.figure(figsize=(15, 17))

    plt.subplot(3, 1, 1)

    red_patch = mpatches.Patch(color='red', label='Amplitude envelope')
    blue_patch = mpatches.Patch(color='blue', label='Waveform')
    green_patch = mpatches.Patch(color='green', label='RMS energy')
    yellow_patch = mpatches.Patch(color='yellow', label='Zero crossing rate')
    plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch], fontsize=10, shadow=True, framealpha=0.5,
               edgecolor='b',
               title='Legend')

    librosa.display.waveshow(audio, sr=sr, alpha=0.5)
    plt.plot(t, ae, color="r")
    plt.plot(t, rms, color="g")
    plt.plot(t, zcr, color="y")
    plt.title(filename)
    plt.ylim((-1, 1))
    plt.savefig(Config.MODEL_PATH + '/' + filename + '_amplitude_envelope_with_rms.png')


def _amplitude_envelope(signal, frame_length, hop_length) -> ndarray:
    result = []
    for i in range(0, len(signal), hop_length):
        current_ae = max(signal[i:i + frame_length])
        result.append(current_ae)
    return np.array(result)


def _rms(signal, frame_length, hop_length) -> ndarray:
    result = []
    for i in range(0, len(signal), hop_length):
        current_rms = np.sqrt(np.sum(signal[i:i + frame_length] ** 2) / frame_length)
        result.append(current_rms)
    return np.array(result)
