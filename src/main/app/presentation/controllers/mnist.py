from flask import Blueprint, jsonify, request

from business.image.classification.mnist.mnist import train, predict

mnist_blueprint = Blueprint('mnist_blueprint', __name__)


@mnist_blueprint.route('/mnist/classify', methods=['POST'])
def predict_img():
    return jsonify(result=predict(request.files.get('image', '')).serialize())


@mnist_blueprint.route('/mnist/train', methods=['POST'])
def train_model():
    return jsonify(result=train('accuracy').serialize())
