from io import StringIO
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import pickle
import os
from datetime import datetime
from app.utils.supabase_client import supabase

def forecast_from_csv(file, sku_id, incremental=True):
    """
    Forecasting with user-specific incremental learning capability
    
    Parameters:
    -----------
    file : file-like object or string
        CSV file containing historical sales data
    sku_id : int
        Product SKU identifier
    user_id : str
        ID of the user making the forecast request
    incremental : bool, default=True
        Whether to use incremental learning (load previous model if available)
    
    Returns:
    --------
    dict
        Forecast results and status information
    """
    try:
        # 1. File handling
        content = file.read().decode('utf-8') if hasattr(file, 'read') else file
        df = pd.read_csv(StringIO(content))
        
        # 2. Data validation
        # required_cols = {'week', 'sku_id', 'units_sold'}
        required_cols = {'week', 'sku_id'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")

        # 3. Data preparation
        df['week'] = pd.to_datetime(df['week'], errors='coerce')
        if df['week'].isnull().any():
            raise ValueError("Invalid date format in 'week' column")
            
        sku_data = df[df['sku_id'] == int(sku_id)].copy()
        if len(sku_data) < 4:  # Minimum 4 weeks of data
            raise ValueError(f"Not enough historical data for SKU {sku_id} (need at least 4 weeks)")
        
        # 4. Prepare time series
        weekly = sku_data.groupby('week', as_index=False)['units_sold'].sum()
        weekly.columns = ['ds', 'y']
        
        # 5. Set up user-specific model storage paths
        model_dir = os.path.join('app', 'models', 'prophet')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'prophet_model_sku_{sku_id}.pkl')
        
        # 6. Incremental learning - load existing user-specific model if available
        model = None
        previous_data = None
        is_new_model = True
        
        if incremental and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    saved_model_data = pickle.load(f)
                    model = saved_model_data.get('model')
                    previous_data = saved_model_data.get('data')
                    is_new_model = False
                    print(f"Loaded existing model for user, SKU {sku_id}")
            except Exception as e:
                print(f"Error loading existing model: {str(e)}. Creating new model.")
                model = None
        
        # 7. Merge new data with previous user data if available
        if previous_data is not None:
            # Identify new data points not in previous dataset
            previous_dates = set(previous_data['ds'].dt.strftime('%Y-%m-%d'))
            new_data = weekly[~weekly['ds'].dt.strftime('%Y-%m-%d').isin(previous_dates)]
            
            # Only update if there's actually new data
            if len(new_data) > 0:
                combined_data = pd.concat([previous_data, new_data]).reset_index(drop=True)
                print(f"Added {len(new_data)} new data points to user's model")
            else:
                combined_data = previous_data
                print("No new data points to add to model")
        else:
            combined_data = weekly
        
        # 8. Configure and train Prophet model based on user's data
        if model is None:
            # Create new model with appropriate parameters
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                growth='flat',
                changepoint_prior_scale=0.05
            )
            
            # If we have enough data for this user, enable yearly seasonality
            if len(combined_data) >= 52:
                model.add_seasonality(name='yearly', period=52, fourier_order=5)
        
        # 9. Train model with all available user data
        model.fit(combined_data)
        
        # 10. Save the updated user-specific model
        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'data': combined_data,
                    # 'user_id': user_id,
                    'last_updated': datetime.now().isoformat()
                }, f)
            model_saved = True
        except Exception as save_err:
            print(f"Error saving model: {str(save_err)}")
            model_saved = False
        
        # 11. Generate forecast (4 weeks)
        future = model.make_future_dataframe(periods=4, freq='W')
        forecast = model.predict(future)
        
        # 12. Prepare forecast results
        forecast_results = []
        for _, row in forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(4).iterrows():
            # Ensure no negative forecasts
            forecast_value = max(0, float(round(row['yhat'], 2)))
            lower_bound = max(0, float(round(row['yhat_lower'], 2)))
            upper_bound = max(0, float(round(row['yhat_upper'], 2)))
            
            forecast_results.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'forecast': forecast_value,
                'forecast_lower': lower_bound,
                'forecast_upper': upper_bound
            })
        
        # 13. Save forecast to database with user attribution
        db_save_success = False
        db_error_message = None
        
        try:
            # Check if a forecast for this user & SKU already exists
            existing = supabase.table('forecasts').select('*').eq('sku_id', int(sku_id)).execute()
            
            forecast_entry = {
                'sku_id': int(sku_id),
                # 'user_id': user_id,
                'forecast_data': forecast_results,
                'model_metadata': {
                    'is_incremental': incremental,
                    'is_new_model': is_new_model,
                    'data_points': len(combined_data),
                    'model_saved': model_saved
                },
                # 'updated_at': datetime.now().isoformat()
            }
            
            if existing and hasattr(existing, 'data') and len(existing.data) > 0:
                # Update existing forecast record
                response = supabase.table('forecasts').update(forecast_entry).eq('sku_id', int(sku_id)).execute()
            else:
                # Insert new forecast record
                forecast_entry['generated_at'] = datetime.now().isoformat()
                response = supabase.table('forecasts').insert(forecast_entry).execute()
            
            # Verify successful response
            if response and not (hasattr(response, 'error') and response.error):
                db_save_success = True
            else:
                db_error_message = str(response.error) if hasattr(response, 'error') else "Unknown database error"
                
        except Exception as db_err:
            db_error_message = str(db_err)
            print(f"Database error during save: {db_error_message}")
        
        # 14. Build response with detailed status
        result = {
            'sku_id': int(sku_id),
            # 'user_id': user_id,
            'forecast': forecast_results,
            'db_saved': db_save_success,
            'model_info': {
                'is_incremental': incremental,
                'is_new_model': is_new_model,
                'data_points_used': len(combined_data),
                'model_saved': model_saved
            },
            'status': 'success'
        }
        
        if not db_save_success:
            result['db_error'] = db_error_message
            
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'status': 'error'
        }

