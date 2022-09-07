from flask import Blueprint, jsonify, request

from business.mnist.start import train, predict

ml = Blueprint('ml', __name__)


@ml.route('/predict', methods=['POST'])
def predict_img():
    # TODO: Get image from request and pass it as a parameter
    result = predict(request.files.get('image', ''))
    return jsonify(result=result.serialize())


@ml.route('/train', methods=['POST'])
def train_model():
    result = train('accuracy')
    return jsonify(result=result.serialize())