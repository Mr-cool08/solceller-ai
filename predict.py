import torch
import numpy as np
import joblib
from datetime import datetime, timedelta
import requests
import torch.nn as nn
import os
import json

MODEL_DIR = 'models'

class SolarNet(nn.Module):
    def __init__(self, input_size):
        super(SolarNet, self).__init__()
        # Basic architecture
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x, eval_mode=False):
        x = self.relu(self.fc1(x))
        x = self.dropout(x) if not eval_mode else x
        x = self.relu(self.fc2(x))
        x = self.dropout(x) if not eval_mode else x
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def get_radiation_forecast(lat=61.7273, lon=17.1066):
    """Get UV index forecast from OpenMeteo"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat,
        'longitude': lon,
        'timezone': 'Europe/Stockholm',
        'hourly': 'uv_index',
        'forecast_days': 2  # Get tomorrow's data
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Get tomorrow's hourly UV index values
            tomorrow_start = 24  # Skip first 24 hours (today)
            tomorrow_end = 48    # Next 24 hours (tomorrow)
            uv_values = data['hourly']['uv_index'][tomorrow_start:tomorrow_end]
            
            # Print hourly UV values for verification
            print("\nHourly UV index forecast for tomorrow:")
            for hour, uv in enumerate(uv_values):
                if uv > 0:  # Only show hours with UV activity
                    print(f"{hour:02d}:00 - UV Index: {uv:.1f}")
            
            # Calculate average UV index only for hours with significant UV (> 0.1)
            significant_uv = [uv for uv in uv_values if uv > 0.1]
            if significant_uv:
                avg_uv = sum(significant_uv) / len(significant_uv)
                max_uv = max(uv_values)
                print(f"\nUV Index Summary:")
                print(f"Maximum UV: {max_uv:.1f}")
                print(f"Average UV (during active hours): {avg_uv:.1f}")
                
                # Convert UV index to MJ/m² using correct conversion factor
                # 1 UV Index ≈ 0.025 MJ/m² per hour
                # Sum up hourly radiation for the day
                daily_radiation = sum(uv * 0.025 for uv in uv_values)
                print(f"Estimated radiation equivalent: {daily_radiation:.2f} MJ/m²")
                return daily_radiation
            return 0
    except Exception as e:
        print(f"Error getting UV data from OpenMeteo: {e}")
    return None

def get_smhi_forecast(lat=61.7273, lon=17.1066):
    """Get weather forecast from SMHI API"""
    # First get daylight hours data from OpenMeteo
    tomorrow = (datetime.now() + timedelta(days=1)).date()
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat,
        'longitude': lon,
        'timezone': 'Europe/Stockholm',
        'daily': ['sunrise', 'sunset'],
        'start_date': tomorrow.strftime('%Y-%m-%d'),
        'end_date': tomorrow.strftime('%Y-%m-%d'),
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            sunrise = datetime.fromisoformat(data['daily']['sunrise'][0])
            sunset = datetime.fromisoformat(data['daily']['sunset'][0])
            daylight_hours = (sunset - sunrise).total_seconds() / 3600
        else:
            print("Could not get daylight hours data")
            daylight_hours = None
    except:
        print("Error getting daylight hours data")
        daylight_hours = None

    base_url = f"https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/geotype/point/lon/{lon}/lat/{lat}/data.json"
    
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            tomorrow = (datetime.now() + timedelta(days=1)).date()
            
            # Initialize weather parameters
            weather_data = {
                'max_temp': float('-inf'),
                'min_temp': float('inf'),
                'precipitation': 0.0,
                'cloud_cover_sum': 0.0,
                'cloud_cover_count': 0,
                'snow': 0.0,
                'rain': 0.0,
                'high_intensity': False,
                'precip_type': None,
                'current_temp': None
            }
            
            # Process forecast data
            parameter_names = set()  # Keep track of all available parameters
            for forecast in data['timeSeries']:

                forecast_time = datetime.fromisoformat(forecast['validTime'].replace('Z', '+00:00'))
                if forecast_time.date() == tomorrow:
                    # Collect parameter names
                    for parameter in forecast['parameters']:
                        parameter_names.add(parameter['name'])
                    
                    # Extract parameters for each time point
                    for parameter in forecast['parameters']:
                        param_name = parameter['name']
                        value = parameter['values'][0]
                        
                        if param_name == 't':  # Air temperature (Parameter 1)
                            weather_data['max_temp'] = max(weather_data['max_temp'], value)
                            weather_data['min_temp'] = min(weather_data['min_temp'], value)
                            weather_data['current_temp'] = value
                        elif param_name == 'pis':  # Precipitation intensity (Parameter 15)
                            # Track high intensity precipitation for better snow/rain determination
                            if value > 0:
                                weather_data['high_intensity'] = True
                        elif param_name == 'pcat':  # Precipitation category (Parameter 17/18)
                            # Store precipitation type for this time period
                            weather_data['precip_type'] = value
                        elif param_name == 'pmean':  # Hourly precipitation (Parameter 7)
                            # Only add precipitation during the actual forecast hour
                            if value > 0:
                                weather_data['precipitation'] += value
                                # Determine if it's rain or snow based on temperature and category
                                temp = weather_data.get('current_temp', 0)
                                if temp <= 0 or weather_data.get('precip_type') in [1, 2]:  # Snow categories
                                    weather_data['snow'] += value
                                else:
                                    weather_data['rain'] += value
                        elif param_name == 'tcc_mean':  # Total cloud cover (Parameter 16)
                            # SMHI reports this as 0-100%, invert it since higher cloud cover means less sun
                            value = 100 - value  # Invert the value
                            hour = forecast_time.hour
                            # Only count daylight hours (6:00-18:00) with higher weight
                            if 6 <= hour <= 18:
                                weight = 2.0
                                weather_data['cloud_cover_sum'] += value * weight
                                weather_data['cloud_cover_count'] += weight
                                print(f"Cloud cover at {forecast_time.strftime('%H:%M')}: {value}% clear sky (weighted)")
                            else:
                                # Still track nighttime for completeness but with lower weight
                                weather_data['cloud_cover_sum'] += value
                                weather_data['cloud_cover_count'] += 1
                                print(f"Cloud cover at {forecast_time.strftime('%H:%M')}: {value}% clear sky")
                        elif param_name == 'wsymb':  # Weather symbol (for precipitation type)
                            # SMHI wsymb categories for snow: 12,13,14,15,16,17,18,19
                            if value in [12,13,14,15,16,17,18,19]:  # Snow or Snow+Rain
                                weather_data['snow'] += weather_data['precipitation']
                            else:  # Assume rain for all other precipitation
                                weather_data['rain'] += weather_data['precipitation']
            
            # Print available parameters for debugging
            print("\nAvailable SMHI parameters:", sorted(parameter_names))
            
            # Calculate averages and convert units
            if weather_data['max_temp'] == float('-inf'):
                print("No temperature data available in SMHI forecast")
                return None
            
            # Ensure we have both max and min temperatures
            if weather_data['min_temp'] == float('inf'):
                weather_data['min_temp'] = weather_data['max_temp']
            
            # Calculate weighted average cloud cover
            if weather_data['cloud_cover_count'] > 0:
                cloud_cover = weather_data['cloud_cover_sum'] / weather_data['cloud_cover_count']
                print(f"\nWeighted average clear sky: {cloud_cover:.1f}% (weighted more heavily during daylight hours)")
            else:
                cloud_cover = 0
                print("\nWarning: No cloud cover data available")
            
            # Get UV index forecast instead of radiation
            radiation_proxy = get_radiation_forecast(lat, lon)
            if radiation_proxy is None:
                print("Warning: Could not get UV index data from OpenMeteo")
                radiation_proxy = 0
            
            # Ensure precipitation values are not negative
            weather_data['precipitation'] = max(0, weather_data['precipitation'])
            weather_data['rain'] = max(0, weather_data['rain'])
            weather_data['snow'] = max(0, weather_data['snow'])
            
            # If we have precipitation but no rain/snow classification, assume it's rain
            if weather_data['precipitation'] > 0 and weather_data['rain'] == 0 and weather_data['snow'] == 0:
                weather_data['rain'] = weather_data['precipitation']
            
            print("SMHI Forecast data retrieved successfully")
            print(f"Found {weather_data['cloud_cover_count']} weather samples for tomorrow")
            
            # Print status of data collection
            print("\nData collection summary:")
            print(f"Cloud cover readings: {weather_data['cloud_cover_count']}")
            print(f"Available parameters: {sorted(parameter_names)}")
            return [
                weather_data['max_temp'],
                weather_data['min_temp'],
                weather_data['precipitation'],
                weather_data['rain'],
                weather_data['snow'],
                min(100, max(0, cloud_cover)),  # Ensure cloud cover is between 0-100%
                radiation_proxy  # Using scaled UV index as radiation proxy
                # Removed daylight_hours since model wasn't trained with it
            ]
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching SMHI forecast: {str(e)}")
    except Exception as e:
        print(f"Error processing SMHI forecast data: {str(e)}")
    return None

def get_model_versions():
    """Get all available model versions"""
    version_file = os.path.join(MODEL_DIR, 'model_versions.json')
    try:
        with open(version_file, 'r') as f:
            versions = json.load(f)
        return versions
    except FileNotFoundError:
        return None

def select_model_version():
    """Interactive model version selection"""
    versions = get_model_versions()
    if not versions:
        print("No trained models found")
        return None
        
    print("\nAvailable model versions:")
    for v in sorted(versions, key=lambda x: x['version']):
        print(f"\nVersion {v['version']} - {v['timestamp']}")
        print(f"MAE: {v['mae']:.2f} kWh")
        print(f"RMSE: {v['rmse']:.2f} kWh")
        print(f"Validation Loss: {v['val_loss']:.4f}")
        
    while True:
        try:
            choice = input("\nSelect model version (or press Enter for latest): ").strip()
            if not choice:  # Empty input - use latest
                return max(versions, key=lambda x: x['version'])
            
            version_num = int(choice)
            selected = next((v for v in versions if v['version'] == version_num), None)
            if selected:
                return selected
            print(f"Version {version_num} not found")
        except ValueError:
            print("Please enter a valid version number")

def get_latest_model_version():
    """Get the latest model version from model_versions.json"""
    version_file = os.path.join(MODEL_DIR, 'model_versions.json')
    try:
        with open(version_file, 'r') as f:
            versions = json.load(f)
        if versions:
            latest = max(versions, key=lambda x: x['version'])
            return latest
        return None
    except FileNotFoundError:
        return None

def predict_solar_output():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Get user-selected model version
        version = select_model_version()
        if version is None:
            raise FileNotFoundError("No trained models found")
            
        print(f"\nUsing model version {version['version']}:")
        print(f"MAE: {version['mae']:.2f} kWh")
        print(f"RMSE: {version['rmse']:.2f} kWh")
        
        # Load model and scalers using version paths
        model_path = os.path.join(MODEL_DIR, version['model_file'])
        feature_scaler_path = os.path.join(MODEL_DIR, version['feature_scaler'])
        target_scaler_path = os.path.join(MODEL_DIR, version['target_scaler'])
        
        # Load the trained model
        checkpoint = torch.load(model_path, map_location=device)
        model = SolarNet(checkpoint['input_size']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load the scalers
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        
        # Get weather forecast from SMHI
        weather_data = get_smhi_forecast()
        
        if weather_data:
            # Scale the features
            weather_data_scaled = feature_scaler.transform([weather_data])
            
            # Convert to tensor and predict
            with torch.no_grad():
                input_tensor = torch.FloatTensor(weather_data_scaled).to(device)
                scaled_prediction = model(input_tensor, eval_mode=True).cpu().numpy()
                
                # Inverse transform the prediction
                prediction = target_scaler.inverse_transform(scaled_prediction)[0][0]
            
            tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"\nPredicted solar energy production for {tomorrow}: {prediction:.2f} kWh")
            print("\nWeather conditions used for prediction:")
            print(f"Max Temperature: {weather_data[0]:.1f}°C")
            print(f"Min Temperature: {weather_data[1]:.1f}°C")
            print(f"Precipitation: {weather_data[2]:.1f}mm")
            print(f"Rain: {weather_data[3]:.1f}mm")
            print(f"Snowfall: {weather_data[4]:.1f}mm")
            print(f"Cloud Cover: {weather_data[5]:.0f}%")
            print(f"Solar Radiation: {weather_data[6]:.2f} MJ/m²")
            print(f"Daylight Hours: {weather_data[7]:.2f}h")
            return prediction
        else:
            print("Could not fetch weather forecast data from SMHI")
    except FileNotFoundError as e:
        print("Error: Model files not found. Please train the model first using train_model.py")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Make sure you have trained the model first using train_model.py")
    return None    

if __name__ == "__main__":
    predict_solar_output()