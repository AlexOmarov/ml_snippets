from phonemizer.backend import EspeakBackend
from pymorphy2 import MorphAnalyzer
from werkzeug.datastructures import FileStorage

from business.audio.generation.preprocessing.audio_retrieval import retrieve_via_storage
from business.audio.generation.preprocessing.preprocessing import get_phonemes, get_feature_vector
from presentation.api.audio_generation_request import AudioGenerationRequest
from presentation.api.audio_generation_result import AudioGenerationResult
from src.main.resource.config import Config


def generate(audio: FileStorage,
             request: AudioGenerationRequest,
             morph: MorphAnalyzer,
             backend: EspeakBackend) -> AudioGenerationResult:
    audio, sr = retrieve_via_storage(audio, Config.AG_SAMPLE_RATE)
    phonemes = get_phonemes(request.text, "", morph, backend)
    feature_vector = get_feature_vector(audio, Config.AG_FRAME_LENGTH, Config.AG_HOP_LENGTH, sr,
                                        num_mels=Config.AG_NUM_MELS)
    return AudioGenerationResult("")
