from werkzeug.datastructures import FileStorage

from presentation.api.audio_generation_request import AudioGenerationRequest
from presentation.api.audio_generation_result import AudioGenerationResult


def generate(audio: FileStorage, request: AudioGenerationRequest) -> AudioGenerationResult:
    path = ""
    return AudioGenerationResult(path)
