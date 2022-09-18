from flask import Blueprint, jsonify, request

from business.audio.classification.sperctrogram.spectr import train, predict

audio_blueprint = Blueprint('audio_blueprint', __name__)


@audio_blueprint.route('/audio/classify', methods=['POST'])
def predict_img():
    return jsonify(result=predict(request.files.get('audio', '')).serialize())


@audio_blueprint.route('/audio/train', methods=['POST'])
def train_model():
    return jsonify(result=train('accuracy').serialize())
