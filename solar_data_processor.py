import pandas as pd
import os
import requests
from datetime import datetime
import json
from io import StringIO  # Import StringIO from io module

def process_solar_file(file_path):
    try:
        # Read all lines to skip the units row properly
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if len(lines) < 3:
            print(f"File {file_path} has insufficient data (less than 3 lines)")
            return None
            
        # Skip empty lines and get header and data
        header = None
        data_line = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            if header is None:
                header = line
            elif data_line is None and i >= 2:  # Skip the units row (line 1)
                data_line = line
                break
        
        if not header or not data_line:
            print(f"Could not find valid header and data in {file_path}")
            return None
            
        # Create a temporary CSV with just header and data
        csv_content = header + '\n' + data_line
        
        # Read the CSV using pandas with StringIO from io module
        df = pd.read_csv(StringIO(csv_content))
        
        # Get the first (and only) data row
        values = df.iloc[0]
        
        try:
            # Extract values using the full column names
            data = {
                'date': values['Datum och tid'],
                'inverter1': float(values['Energi per växelriktare | Symo Advanced 20.0-3-M (1)']),
                'inverter2': float(values['Energi per växelriktare | Symo GEN24 10.0 Plus']),
                'inverter1_kwp': float(values['Energi per växelriktare per kWp | Symo Advanced 20.0-3-M (1)']),
                'inverter2_kwp': float(values['Energi per växelriktare per kWp | Symo GEN24 10.0 Plus']),
                'total': float(values['Total anläggning'])
            }
            return data
        except KeyError as e:
            print(f"Missing column in {file_path}: {e}")
            print("Available columns:", df.columns.tolist())
            return None
        except ValueError as e:
            print(f"Error converting values in {file_path}: {e}")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def get_weather_data(date, lat=61.7273, lon=17.1066):
    """Get weather data from Open-Meteo API"""
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date.strftime('%Y-%m-%d'),
        "end_date": date.strftime('%Y-%m-%d'),
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", 
                 "rain_sum", "snowfall_sum", "cloudcover_mean", "shortwave_radiation_sum",
                 "sunrise", "sunset"],
        "timezone": "Europe/Stockholm"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            daily_data = data['daily']
            
            # Calculate daylight hours
            sunrise = datetime.fromisoformat(daily_data['sunrise'][0])
            sunset = datetime.fromisoformat(daily_data['sunset'][0])
            daylight_hours = (sunset - sunrise).total_seconds() / 3600  # Convert to hours
                
            return [
                daily_data['temperature_2m_max'][0],
                daily_data['temperature_2m_min'][0],
                daily_data['precipitation_sum'][0],
                daily_data['rain_sum'][0],
                daily_data['snowfall_sum'][0],
                daily_data['cloudcover_mean'][0],
                daily_data['shortwave_radiation_sum'][0],
                daylight_hours
            ]
    except requests.exceptions.RequestException as e:
        print(f"Network error getting weather data: {e}")
    except Exception as e:
        print(f"Error processing weather data: {e}")
    return None

def main():
    # Create output directory if it doesn't exist
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all solar data files
    solar_data_dir = "solardata"
    combined_data = []
    error_count = 0
    success_count = 0
    
    print("\nStarting data processing...")
    
    for file in sorted(os.listdir(solar_data_dir)):
        if file.endswith('.csv'):
            file_path = os.path.join(solar_data_dir, file)
            print(f"\nProcessing {file}...")
            
            # Extract solar data
            solar_data = process_solar_file(file_path)
            if solar_data is None:
                error_count += 1
                continue
            
            # Get weather data for the same date
            try:
                date = datetime.strptime(solar_data['date'], '%d.%m.%Y')
                print(f"Getting weather data for {date.strftime('%Y-%m-%d')}...")
                weather_data = get_weather_data(date)
                
                if (weather_data):
                    # Combine solar and weather data
                    combined_entry = {**solar_data}
                    combined_entry.update({
                        'max_temp': weather_data[0],
                        'min_temp': weather_data[1],
                        'precipitation': weather_data[2],
                        'rain': weather_data[3],
                        'snowfall': weather_data[4],
                        'cloud_cover': weather_data[5],
                        'solar_radiation': weather_data[6],
                        'daylight_hours': weather_data[7]
                    })
                    combined_data.append(combined_entry)
                    success_count += 1
                    print(f"Successfully processed {file}")
                else:
                    print(f"No weather data available for {file}")
                    error_count += 1
            except ValueError as e:
                print(f"Error parsing date in {file}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing {file}: {e}")
                error_count += 1
    
    # Save to CSV file
    if combined_data:
        df = pd.DataFrame(combined_data)
        output_file = os.path.join(output_dir, 'combined_solar_weather_data.csv')
        df.to_csv(output_file, index=False)
        print(f"\nProcessing complete:")
        print(f"Successfully processed: {success_count} files")
        print(f"Failed to process: {error_count} files")
        print(f"Data saved to: {output_file}")
    else:
        print("\nNo data was processed successfully")

if __name__ == "__main__":
    main()