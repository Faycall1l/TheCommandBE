# from io import StringIO
# import pandas as pd
# from prophet import Prophet
# from prophet.plot import plot_plotly
# from app.utils.supabase_client import supabase

# def forecast_from_csv(file, sku_id):
#     """
#     Improved forecasting with better seasonality handling and data validation
#     """
#     try:
#         # 1. File handling
#         content = file.read().decode('utf-8') if hasattr(file, 'read') else file
#         df = pd.read_csv(StringIO(content))
        
#         # 2. Data validation
#         required_cols = {'week', 'sku_id', 'units_sold'}
#         if not required_cols.issubset(df.columns):
#             missing = required_cols - set(df.columns)
#             raise ValueError(f"Missing columns: {missing}")

#         # 3. Data preparation
#         df['week'] = pd.to_datetime(df['week'], errors='coerce')
#         if df['week'].isnull().any():
#             raise ValueError("Invalid date format in 'week' column")
            
#         sku_data = df[df['sku_id'] == int(sku_id)].copy()
#         if len(sku_data) < 4:  # Minimum 4 weeks of data
#             raise ValueError(f"Not enough historical data for SKU {sku_id} (need at least 4 weeks)")
        
#         # 4. Prepare time series
#         weekly = sku_data.groupby('week', as_index=False)['units_sold'].sum()
#         weekly.columns = ['ds', 'y']
        
#         # 5. Configure Prophet with constrained growth
#         model = Prophet(
#             yearly_seasonality=False,  # Disable until we have >1 year data
#             weekly_seasonality=True,
#             daily_seasonality=False,
#             growth='flat',  # Prevents unrealistic trends with limited data
#             changepoint_prior_scale=0.05  # Reduce trend flexibility
#         )
        
#         # 6. Train model
#         model.fit(weekly)
        
#         # 7. Generate conservative forecast (2 weeks)
#         future = model.make_future_dataframe(periods=2, freq='W')
#         forecast = model.predict(future)
        
#         # 8. Prepare realistic output
#         forecast_results = []
#         for _, row in forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(2).iterrows():
#             # Ensure no negative forecasts
#             forecast_value = max(0, float(round(row['yhat'], 2)))
#             lower_bound = max(0, float(round(row['yhat_lower'], 2)))
#             upper_bound = max(0, float(round(row['yhat_upper'], 2)))
            
#             forecast_results.append({
#                 'date': row['ds'].strftime('%Y-%m-%d'),
#                 'forecast': forecast_value,
#                 'forecast_lower': lower_bound,
#                 'forecast_upper': upper_bound
#             })
        
#         # 9. Save to database
#         db_save_success = False
#         try:
#             response = supabase.table('forecasts').insert({
#                 'sku_id': int(sku_id),
#                 'forecast_data': forecast_results
#             }).execute()
#             db_save_success = True if not response.error else False
#         except Exception as db_error:
#             print(f"Database error: {str(db_error)}")
        
#         return {
#             'sku_id': int(sku_id),
#             'forecast': forecast_results,
#             'db_saved': db_save_success,
#             'status': 'success'
#         }
        
#     except Exception as e:
#         return {
#             'error': str(e),
#             'status': 'error'
#         }


from io import StringIO
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from app.utils.supabase_client import supabase

def forecast_from_csv(file, sku_id):
    """
    Improved forecasting with better seasonality handling and data validation
    and reliable database saving
    """
    try:
        # 1. File handling
        content = file.read().decode('utf-8') if hasattr(file, 'read') else file
        df = pd.read_csv(StringIO(content))
        
        # 2. Data validation
        required_cols = {'week', 'sku_id', 'units_sold'}
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
        
        # 5. Configure Prophet with constrained growth
        model = Prophet(
            yearly_seasonality=False,  # Disable until we have >1 year data
            weekly_seasonality=True,
            daily_seasonality=False,
            growth='flat',  # Prevents unrealistic trends with limited data
            changepoint_prior_scale=0.05  # Reduce trend flexibility
        )
        
        # 6. Train model
        model.fit(weekly)
        
        # 7. Generate conservative forecast (2 weeks)
        future = model.make_future_dataframe(periods=2, freq='W')
        forecast = model.predict(future)
        
        # 8. Prepare realistic output
        forecast_results = []
        for _, row in forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(2).iterrows():
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
        
        # 9. Improved database saving with robust error handling
        db_save_success = False
        db_error_message = None
        
        try:
            # Check if a forecast for this SKU already exists
            existing = supabase.table('forecasts').select('*').eq('sku_id', int(sku_id)).execute()
            
            if existing and hasattr(existing, 'data') and len(existing.data) > 0:
                # Update existing forecast record
                response = supabase.table('forecasts').update({
                    'forecast_data': forecast_results,
                    'updated_at': pd.Timestamp.now().isoformat()
                }).eq('sku_id', int(sku_id)).execute()
            else:
                # Insert new forecast record
                response = supabase.table('forecasts').insert({
                    'sku_id': int(sku_id),
                    'forecast_data': forecast_results,
                    'generated_at': pd.Timestamp.now().isoformat()
                }).execute()
            
            # Verify successful response
            if response and not (hasattr(response, 'error') and response.error):
                db_save_success = True
            else:
                db_error_message = str(response.error) if hasattr(response, 'error') else "Unknown database error"
                
        except Exception as db_err:
            db_error_message = str(db_err)
            print(f"Database error during save: {db_error_message}")
        
        # 10. Build response with detailed status
        result = {
            'sku_id': int(sku_id),
            'forecast': forecast_results,
            'db_saved': db_save_success,
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