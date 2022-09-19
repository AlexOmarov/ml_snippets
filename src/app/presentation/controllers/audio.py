from flask import Blueprint, jsonify, request

from business.audio.analysis.analysis import analyse

audio_blueprint = Blueprint('audio_blueprint', __name__)


@audio_blueprint.route('/audio/analyse', methods=['POST'])
def predict_img():
    return jsonify(result=analyse(
        request.files.get('audio', ''),
        request.args.get('frame_length', default=2048, type=int),
        request.args.get('hop_length', default=1024, type=int)
    ).serialize())
