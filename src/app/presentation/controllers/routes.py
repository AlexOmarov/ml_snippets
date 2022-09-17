from flask import Blueprint, jsonify, request

from business.mnist.start import train, predict

ml = Blueprint('ml', __name__)


@ml.route('/predict', methods=['POST'])
def predict_img():
    return jsonify(result=predict(request.files.get('image', '')).serialize())


@ml.route('/train', methods=['POST'])
def train_model():
    return jsonify(result=train('accuracy').serialize())
