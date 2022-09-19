from flask import Blueprint, jsonify, request

from business.audio.classification.sperctrogram.spectr import analyse

audio_blueprint = Blueprint('audio_blueprint', __name__)


@audio_blueprint.route('/audio/analyse', methods=['POST'])
def predict_img():
    return jsonify(result=analyse(request.files.get('audio', '')).serialize())
