from flask import Blueprint, request, jsonify
from app.services.forecasting import forecast_from_csv

forecast_blueprint = Blueprint("forecast", __name__)

@forecast_blueprint.route("/upload", methods=["POST"])
def upload_forecast():
    """
    Improved forecast endpoint with better error handling
    """
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'No file part'}), 400
        
    file = request.files['file']
    sku_id = request.form.get('sku_id')
    
    if not file.filename:
        return jsonify({'status': 'error', 'error': 'No selected file'}), 400
        
    if not sku_id or not sku_id.isdigit():
        return jsonify({'status': 'error', 'error': 'Invalid SKU ID'}), 400
    
    try:
        result = forecast_from_csv(file, sku_id)
        status_code = 200 if result['status'] == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f"Processing error: {str(e)}"
        }), 500