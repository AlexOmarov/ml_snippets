from flask import Blueprint, jsonify, request

from business.audio.analysis.analysis import analyse
from business.audio.classification.classification import predict
from business.audio.classification.readaudio import read
from business.audio.generation.generation import generate
from presentation.api.audio_generation_request import AudioGenerationRequest

audio_blueprint = Blueprint('audio_blueprint', __name__)


@audio_blueprint.route('/audio/analyse', methods=['POST'])
def analyse_audio():
    return jsonify(result=analyse(
        request.files.get('audio', ''),
        request.args.get('frame_length', default=2048, type=int),
        request.args.get('hop_length', default=1024, type=int)
    ).serialize())


@audio_blueprint.route('/audio/classify', methods=['POST'])
def classify():
    return jsonify(result=predict(request.files.get('audio', '')).serialize())


@audio_blueprint.route('/audio/generate', methods=['POST'])
def generate_audio():
    return jsonify(
        result=generate(request.files.get('audio', ''), AudioGenerationRequest(**request.get_json())).serialize()
    )


@audio_blueprint.route('/audio/read', methods=['POST'])
def read_audio():
    return jsonify(result=read(request.files.get('image', '')).serialize())
