from flask import Blueprint, request

ml = Blueprint('ml', __name__)


@ml.route('/start', methods=['POST'])
def recommend():
    return request
