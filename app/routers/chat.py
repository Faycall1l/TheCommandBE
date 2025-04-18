from flask import Blueprint, request, jsonify
from app.services.chat_rag import get_chat_response
from app.utils.cleaner import clean_sales_data
from app.services.forecasting import forecast_from_csv
from io import BytesIO

chat_blueprint = Blueprint("chat", __name__)

@chat_blueprint.route("/ask", methods=["POST"])
def ask_advisor():
    file = request.files.get("file")
    sku_id = request.form.get("sku_id")
    question = request.form.get("question")

    if not file or not sku_id or not question:
        return jsonify({"error": "file, sku_id, and question are required"}), 400

    try:
        file_bytes = BytesIO(file.read())
        history_df = clean_sales_data(file_bytes)
        file_bytes.seek(0)
        forecast_df = forecast_from_csv(file_bytes, sku_id)

        response = get_chat_response(forecast_df, history_df, question)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
