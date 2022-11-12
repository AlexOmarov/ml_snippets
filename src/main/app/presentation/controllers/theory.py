from flask import Blueprint, jsonify, request

from business.theory.basic_tensorflow import first

theory_blueprint = Blueprint('theory_blueprint', __name__)


@theory_blueprint.route('/theory', methods=['POST'])
def analyse_audio():
    return jsonify(result=first(request).serialize())
