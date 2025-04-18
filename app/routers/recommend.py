from flask import Blueprint, request, jsonify
import json
import os

recommend_blueprint = Blueprint("recommend", __name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/trending_data.json')

@recommend_blueprint.route("/recommendations", methods=["GET"])
def get_trends():
    niche = request.args.get("niche", "") #.lower()

    if not niche:
        return jsonify({"error": "Please provide a niche as a query parameter."}), 400

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            trending = json.load(f)

        if niche not in trending:
            return jsonify({"error": f"Niche '{niche}' not found."}), 404

        return jsonify({
            "niche": niche,
            "recommendations": trending[niche]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
