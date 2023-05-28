import librosa
from phonemizer.backend import EspeakBackend
from pymorphy2 import MorphAnalyzer
from werkzeug.datastructures import FileStorage

from business.audio.generation.preprocessing.preprocessing import get_phonemes, get_feature_vector
from presentation.api.audio_generation_request import AudioGenerationRequest
from presentation.api.audio_generation_result import AudioGenerationResult


def generate(audio: FileStorage,
             request: AudioGenerationRequest,
             morph: MorphAnalyzer,
             backend: EspeakBackend) -> AudioGenerationResult:
    audio, sr = librosa.load(audio)
    phonemes = get_phonemes(request.text, "", morph, backend)
    feature_vector = get_feature_vector(audio, frame_length=, hop_length=, sr, num_mels=)
    return AudioGenerationResult(path)
