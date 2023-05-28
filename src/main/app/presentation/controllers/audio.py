from flask import Blueprint, jsonify, request

from business.audio.analysis.analysis import analyse
from business.audio.generation.generation import generate
from business.util.ml_context.context import get_morph, get_espeak_backend
from presentation.api.audio_generation_request import AudioGenerationRequest

audio_blueprint = Blueprint('audio_blueprint', __name__)


@audio_blueprint.route('/audio/analyse', methods=['POST'])
def analyse_audio():
    return jsonify(result=analyse(
        request.files.get('audio', ''),
        request.args.get('frame_length', default=2048, type=int),
        request.args.get('hop_length', default=1024, type=int)
    ).serialize())


@audio_blueprint.route('/audio/generate', methods=['POST'])
def generate_audio():
    return jsonify(
        result=generate(
            request.files.get('audio', ''),
            AudioGenerationRequest(**request.get_json()),
            get_morph(),
            get_espeak_backend("ru-RU")
        ).serialize()
    )
