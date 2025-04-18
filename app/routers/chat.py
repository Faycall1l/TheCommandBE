from flask import Blueprint, request, jsonify
from app.services.chat_rag import get_chat_response
from app.utils.cleaner import clean_sales_data
from app.services.forecasting import forecast_from_csv
from io import BytesIO

chat_blueprint = Blueprint("chat", __name__)

@chat_blueprint.route("/ask", methods=["POST"])
def ask_advisor():
    """Endpoint for sales forecasting and business advice"""
    # Validate required inputs
    if 'file' not in request.files or not request.files['file']:
        return jsonify({"error": "Sales data file is required"}), 400
    
    file = request.files['file']
    sku_id = request.form.get('sku_id')
    question = request.form.get('question')

    if not sku_id or not sku_id.isdigit():
        return jsonify({"error": "Valid SKU ID is required"}), 400
    if not question or len(question.strip()) < 5:
        return jsonify({"error": "Detailed question is required"}), 400

    try:
        # Process file once and reuse
        file_content = file.read()
        
        # Get historical data
        history_df = clean_sales_data(BytesIO(file_content))
        
        # Get forecast
        forecast_result = forecast_from_csv(BytesIO(file_content), int(sku_id))
        if forecast_result.get('status') != 'success':
            return jsonify({
                "error": forecast_result.get('error', 'Forecast generation failed'),
                "details": forecast_result
            }), 400

        # Get AI response
        response = get_chat_response(
            forecast_data=forecast_result['forecast'],
            history_df=history_df,
            question=question
        )

        return jsonify({
            "answer": response,
            "forecast": forecast_result['forecast'],
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": "Processing error",
            "details": str(e)
        }), 500