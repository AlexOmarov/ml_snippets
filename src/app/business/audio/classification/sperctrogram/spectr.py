#  Lib imports
import librosa
from werkzeug.datastructures import FileStorage

#  App imports
from business.util.ml_logger import logger
from presentation.api.audio_analysis_result import AudioAnalysisResult

_log = logger.get_logger(__name__.replace('__', '\''))


def analyse(storage: FileStorage) -> AudioAnalysisResult:
    audio: str = _preprocess_single_image(storage)
    return AudioAnalysisResult("metric", [audio])


def _preprocess_single_image(storage: FileStorage) -> str:
    audio = librosa.load(storage)
    # TODO: process image
    return ""
