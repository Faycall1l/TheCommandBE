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




# from io import StringIO
# import pandas as pd
# import numpy as np
# import pickle
# import os
# from datetime import datetime
# from app.utils.supabase_client import supabase
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# def calculate_rmse(actual_values, forecast_values):
#     if len(actual_values) != len(forecast_values):
#         raise ValueError("Arrays must be of the same length")
#     squared_errors = [(actual - forecast)**2 for actual, forecast in zip(actual_values, forecast_values)]
#     mse = sum(squared_errors) / len(squared_errors)
#     rmse = np.sqrt(mse)
#     return rmse

# def create_sequences(data, seq_length):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:i + seq_length])
#         y.append(data[i + seq_length])
#     return np.array(X), np.array(y)

# def build_lstm_model(seq_length):
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# def forecast_from_csv(file, sku_id, incremental=True, test_file=None):
#     try:
#         content = file.read().decode('utf-8') if hasattr(file, 'read') else file
#         df = pd.read_csv(StringIO(content))
#         test_df = None
#         if test_file:
#             test_content = test_file.read().decode('utf-8') if hasattr(test_file, 'read') else test_file
#             test_df = pd.read_csv(StringIO(test_content))

#         required_cols = {'week', 'sku_id', 'units_sold'}
#         if not required_cols.issubset(df.columns):
#             missing = required_cols - set(df.columns)
#             raise ValueError(f"Missing columns: {missing}")

#         if test_df is not None and not required_cols.issubset(test_df.columns):
#             missing = required_cols - set(test_df.columns)
#             raise ValueError(f"Missing columns in test file: {missing}")

#         df['week'] = pd.to_datetime(df['week'], errors='coerce')
#         if df['week'].isnull().any():
#             raise ValueError("Invalid date format in 'week' column")

#         sku_data = df[df['sku_id'] == int(sku_id)].copy()
#         if len(sku_data) < 8:
#             raise ValueError(f"Not enough historical data for SKU {sku_id} (need at least 8 weeks for LSTM)")

#         test_data = None
#         if test_df is not None:
#             test_df['week'] = pd.to_datetime(test_df['week'], errors='coerce')
#             if test_df['week'].isnull().any():
#                 raise ValueError("Invalid date format in test file 'week' column")
#             test_sku_data = test_df[test_df['sku_id'] == int(sku_id)].copy()
#             if len(test_sku_data) > 0:
#                 test_data = test_sku_data.sort_values('week')
#                 print(f"Found {len(test_data)} test data points for SKU {sku_id}")

#         weekly = sku_data.sort_values('week').groupby('week', as_index=False)['units_sold'].sum()
#         weekly.columns = ['ds', 'y']

#         model_dir = os.path.join('app', 'models', 'lstm')
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(model_dir, f'lstm_model_sku_{sku_id}.h5')
#         scaler_path = os.path.join(model_dir, f'lstm_scaler_sku_{sku_id}.pkl')
#         data_path = os.path.join(model_dir, f'lstm_data_sku_{sku_id}.pkl')

#         previous_data = None
#         is_new_model = True
#         seq_length = 4

#         if incremental and os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(data_path):
#             try:
#                 with open(data_path, 'rb') as f:
#                     saved_data = pickle.load(f)
#                     previous_data = saved_data.get('data')
#                 with open(scaler_path, 'rb') as f:
#                     scaler = pickle.load(f)
#                 is_new_model = False
#                 print(f"Loaded existing model data for SKU {sku_id}")
#             except Exception as e:
#                 print(f"Error loading existing model data: {str(e)}. Creating new model.")
#                 previous_data = None
#         else:
#             scaler = MinMaxScaler(feature_range=(0, 1))

#         if previous_data is not None:
#             previous_dates = set(previous_data['ds'].dt.strftime('%Y-%m-%d'))
#             new_data = weekly[~weekly['ds'].dt.strftime('%Y-%m-%d').isin(previous_dates)]
#             if len(new_data) > 0:
#                 combined_data = pd.concat([previous_data, new_data]).sort_values('ds').reset_index(drop=True)
#                 print(f"Added {len(new_data)} new data points to model")
#             else:
#                 combined_data = previous_data
#                 print("No new data points to add to model")
#         else:
#             combined_data = weekly

#         try:
#             with open(data_path, 'wb') as f:
#                 pickle.dump({
#                     'data': combined_data,
#                     'last_updated': datetime.now().isoformat()
#                 }, f)
#         except Exception as save_err:
#             print(f"Error saving data: {str(save_err)}")

#         combined_data_scaled = scaler.fit_transform(combined_data['y'].values.reshape(-1, 1))
#         X, y = create_sequences(combined_data_scaled, seq_length)
#         X = X.reshape((X.shape[0], X.shape[1], 1))

#         if is_new_model:
#             model = build_lstm_model(seq_length)
#         else:
#             model = load_model(model_path)

#         model.fit(X, y, epochs=50, batch_size=1, verbose=0)

#         model.save(model_path)
#         with open(scaler_path, 'wb') as f:
#             pickle.dump(scaler, f)

#         forecast_input = combined_data_scaled[-seq_length:].reshape((1, seq_length, 1))
#         forecast = model.predict(forecast_input, verbose=0)
#         forecast_units = scaler.inverse_transform(forecast)[0][0]

#         result = {
#             'sku_id': sku_id,
#             'forecast_units_sold': float(forecast_units),
#             'status': 'success'
#         }

#         if test_data is not None and len(test_data) >= 1:
#             test_scaled = scaler.transform(test_data['units_sold'].values.reshape(-1, 1))
#             test_X, test_y = create_sequences(test_scaled, seq_length)
#             test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))
#             forecasted_test = model.predict(test_X, verbose=0)
#             forecasted_units = scaler.inverse_transform(forecasted_test).flatten()
#             actual_units = test_data['units_sold'].values[seq_length:]
#             if len(actual_units) == len(forecasted_units):
#                 rmse = calculate_rmse(actual_units, forecasted_units)
#                 result['rmse'] = rmse
#             else:
#                 print("Length mismatch in RMSE calculation")

#         return result

#     except Exception as e:
#         return {'status': 'error', 'message': str(e)}
