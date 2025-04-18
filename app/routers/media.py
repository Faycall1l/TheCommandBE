from flask import Blueprint, request, jsonify
from app.services.media_planner import generate_media_plan
import json

media_blueprint = Blueprint("media", __name__)

@media_blueprint.route("/plan", methods=["POST"])
def get_media_plan():
    try:
        data = request.get_json()
        forecast_data = data.get("forecast")
        niche = data.get("niche", "general")

        if not forecast_data:
            return jsonify({"error": "Missing forecast data"}), 400

        # Optional: convert forecast JSON into readable text
        text = "\n".join([
            f"{row['ds']} → {int(row['yhat'])} units"
            for row in forecast_data
        ])

        plan = generate_media_plan(forecast_data=text, niche=niche)
        return jsonify({"media_plan": plan})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
