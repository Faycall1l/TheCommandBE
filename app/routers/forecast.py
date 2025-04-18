from flask import Blueprint, request, jsonify
from app.services.forecasting import forecast_from_csv

forecast_blueprint = Blueprint("forecast", __name__)

@forecast_blueprint.route("/upload", methods=["POST"])
def upload_forecast():
    file = request.files.get("file")
    sku_id = request.form.get("sku_id", default="223153")

    if not file:
        return jsonify({"error": "No file provided"}), 400

    try:
        forecast = forecast_from_csv(file, sku_id)
        return jsonify(forecast.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
