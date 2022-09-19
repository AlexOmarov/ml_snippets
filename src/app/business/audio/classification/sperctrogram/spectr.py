#  Lib imports
import librosa
import librosa.display
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
    audio: str = _preprocess_single_image(storage)
    return AudioAnalysisResult("metric", [audio])


def _get_amplitude_envelope(audio, sr, filename: str):
    frame_length = 4096
    hop_length = 2048
    ae = _amplitude_envelope(audio, frame_length, hop_length)
    # TODO: how to get t? What is this time?
    frames = range(0, ae.size)
    t = librosa.frames_to_time(frames, hop_length=hop_length)
    plt.figure(figsize=(15, 17))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio, sr=sr, alpha=0.5)
    plt.plot(t, ae, color="r")
    plt.title(filename)
    plt.ylim((-1, 1))
    plt.savefig(Config.MODEL_PATH + '/' + filename + '_amplitude_envelope.png')


def _preprocess_single_image(storage: FileStorage) -> str:
    audio, sr = librosa.load(storage)
    _get_amplitude_envelope(audio, sr, storage.filename)
    return ""


def _amplitude_envelope(signal, frame_size, hop_size) -> ndarray:
    result = []
    for i in range(0, len(signal), hop_size):
        current_ae = max(signal[i:i + frame_size])
        result.append(current_ae)
    return np.array(result)
