# -*- coding: utf-8 -*-
"""
F1 Weather Enrichment
Adds weather data to F1 race dataset using Open-Meteo Historical Weather API.
"""

import pandas as pd
import requests
import requests_cache
import time
from datetime import datetime, timedelta
import numpy as np
import os

# Configuration
INPUT_FILE = '../data/f1_ultimate_data.csv'
OUTPUT_FILE = '../data/f1_complete_dataset.csv'
CACHE_DB = '../cache/weather_cache'

API_URL = 'https://archive-api.open-meteo.com/v1/archive'
API_DELAY = 0.2
DEFAULT_RACE_TIME = "14:00:00"

requests_cache.install_cache(CACHE_DB, backend='sqlite', expire_after=86400)

print("="*80)
print("F1 Weather Data Enrichment")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def parse_race_time(time_str):
    """Parse race time string to hour value, return default if invalid."""
    try:
        if pd.isna(time_str) or not time_str:
            return 14
        
        time_parts = str(time_str).split(':')
        hour = int(time_parts[0])
        
        if len(time_parts) > 1:
            minute = int(time_parts[1])
            if minute >= 30:
                hour += 1
        
        return hour
    except:
        return 14


def get_race_date_from_f1db(year, round_num):
    """Fetch race date from F1DB file."""
    try:
        races_df = pd.read_csv('../data/f1db-csv/f1db-races.csv')
        race = races_df[(races_df['year'] == year) & (races_df['round'] == round_num)]
        
        if not race.empty:
            return race.iloc[0]['date']
        return None
    except Exception as e:
        print(f"  ! Could not fetch date from F1DB: {e}")
        return None


def fetch_weather_data(latitude, longitude, race_date, race_hour):
    """Fetch weather data from Open-Meteo API for specific location and date."""
    try:
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': race_date,
            'end_date': race_date,
            'hourly': 'temperature_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m',
            'timezone': 'UTC'
        }
        
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        from_cache = getattr(response, 'from_cache', False)
        cache_info = " (Cache)" if from_cache else " (API)"
        
        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        
        target_time = f"{race_date}T{race_hour:02d}:00"
        
        if target_time in times:
            idx = times.index(target_time)
            
            weather = {
                'WeatherTemp': hourly.get('temperature_2m', [])[idx],
                'WeatherPrecipitation': hourly.get('precipitation', [])[idx],
                'WeatherPressure': hourly.get('surface_pressure', [])[idx],
                'WeatherWindSpeed': hourly.get('wind_speed_10m', [])[idx],
                'WeatherWindDirection': hourly.get('wind_direction_10m', [])[idx]
            }
            
            return weather, cache_info
        else:
            for i, t in enumerate(times):
                if race_hour <= int(t.split('T')[1].split(':')[0]):
                    weather = {
                        'WeatherTemp': hourly.get('temperature_2m', [])[i],
                        'WeatherPrecipitation': hourly.get('precipitation', [])[i],
                        'WeatherPressure': hourly.get('surface_pressure', [])[i],
                        'WeatherWindSpeed': hourly.get('wind_speed_10m', [])[i],
                        'WeatherWindDirection': hourly.get('wind_direction_10m', [])[i]
                    }
                    return weather, cache_info
            
            return None, cache_info
            
    except requests.exceptions.RequestException as e:
        print(f"    X API Error: {e}")
        return None, ""
    except Exception as e:
        print(f"    X Data Processing Error: {e}")
        return None, ""


def main():
    print("Loading data...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"X ERROR: {INPUT_FILE} not found!")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"  * {len(df):,} rows, {len(df.columns)} columns loaded\n")
    
    initial_cols = len(df.columns)
    
    print("Identifying unique races...")
    
    races_df = pd.read_csv('../data/f1db-csv/f1db-races.csv')
    races_df = races_df[['year', 'round', 'date']].rename(columns={'date': 'RaceDate'})
    
    df = df.merge(races_df, left_on=['Year', 'Round'], right_on=['year', 'round'], how='left')
    df = df.drop(columns=['year', 'round'])
    
    if 'RaceTime' not in df.columns:
        df['RaceTime'] = DEFAULT_RACE_TIME
        print(f"  ! RaceTime column missing, using default ({DEFAULT_RACE_TIME})")
    
    unique_races = df[['Year', 'Round', 'CircuitLat', 'CircuitLng', 'RaceDate', 'RaceTime']].drop_duplicates()
    unique_races = unique_races.dropna(subset=['CircuitLat', 'CircuitLng', 'RaceDate'])
    
    print(f"  * {len(unique_races)} unique races found\n")
    
    print("="*80)
    print("Fetching Weather Data")
    print("="*80 + "\n")
    
    weather_data = []
    success_count = 0
    fail_count = 0
    
    for idx, race in unique_races.iterrows():
        year = race['Year']
        round_num = race['Round']
        lat = race['CircuitLat']
        lng = race['CircuitLng']
        race_date = race['RaceDate']
        race_time = race['RaceTime']
        
        try:
            race_date_obj = pd.to_datetime(race_date)
            race_date_str = race_date_obj.strftime('%Y-%m-%d')
        except:
            print(f"  ! {year} R{round_num}: Invalid date, skipping...")
            fail_count += 1
            continue
        
        race_hour = parse_race_time(race_time)
        
        progress = ((idx + 1) / len(unique_races)) * 100
        print(f"[{progress:>5.1f}%] {year} R{round_num:>2} @ {race_date_str} {race_hour:02d}:00 | ", end='')
        
        weather, cache_info = fetch_weather_data(lat, lng, race_date_str, race_hour)
        
        if weather:
            weather['Year'] = year
            weather['Round'] = round_num
            weather_data.append(weather)
            success_count += 1
            print(f"* Temp: {weather['WeatherTemp']:.1f}C, Precip: {weather['WeatherPrecipitation']:.1f}mm{cache_info}")
        else:
            fail_count += 1
            print(f"X Failed")
        
        if not cache_info:
            time.sleep(API_DELAY)
    
    print("\n" + "="*80)
    print(f"* Success: {success_count}/{len(unique_races)}")
    print(f"X Failed: {fail_count}/{len(unique_races)}")
    print("="*80 + "\n")
    
    print("Merging weather data...")
    
    if weather_data:
        weather_df = pd.DataFrame(weather_data)
        
        df = df.merge(weather_df, on=['Year', 'Round'], how='left')
        
        added_cols = len(df.columns) - initial_cols
        print(f"  * {added_cols} columns added")
        
        print("\nWeather Data Coverage:")
        for col in ['WeatherTemp', 'WeatherPrecipitation', 'WeatherPressure', 'WeatherWindSpeed', 'WeatherWindDirection']:
            if col in df.columns:
                non_null = df[col].notna().sum()
                coverage = (non_null / len(df)) * 100
                print(f"  - {col:<25} : {non_null:>7,} / {len(df):>7,} ({coverage:>5.1f}%)")
    else:
        print("  ! No weather data added!")
    
    print("\nCleaning temporary columns...")
    
    columns_to_drop = ['RaceDate', 'RaceTime']
    existing_drops = [col for col in columns_to_drop if col in df.columns]
    
    if existing_drops:
        df = df.drop(columns=existing_drops)
        print(f"  * {len(existing_drops)} columns removed: {', '.join(existing_drops)}")
    
    print("\n" + "="*80)
    print("Saving File")
    print("="*80)
    
    df.to_csv(OUTPUT_FILE, index=False)
    file_size = os.path.getsize(OUTPUT_FILE) / (1024*1024)
    
    print(f"  * File saved: {OUTPUT_FILE}")
    print(f"  * File size: {file_size:.2f} MB")
    print(f"  * Final dimensions: {len(df):,} rows x {len(df.columns)} columns")
    
    print("\n" + "="*80)
    print("Weather Enrichment Summary")
    print("="*80)
    print(f"  - Total races         : {len(unique_races)}")
    print(f"  - Successful requests : {success_count}")
    print(f"  - Failed requests     : {fail_count}")
    print(f"  - Initial columns     : {initial_cols}")
    print(f"  - Final columns       : {len(df.columns)}")
    print(f"  - Added columns       : {len(df.columns) - initial_cols - len(existing_drops)}")
    
    print("\nSample Weather Data (First 5 Rows):")
    weather_cols = ['Year', 'Round', 'Circuit', 'WeatherTemp', 'WeatherPrecipitation', 
                    'WeatherPressure', 'WeatherWindSpeed', 'WeatherWindDirection']
    available_cols = [col for col in weather_cols if col in df.columns]
    
    if available_cols:
        print(df[available_cols].head(5).to_string(index=False))
    
    print("\n" + "="*80)
    print("Weather Enrichment Complete")
    print(f"File: {os.path.abspath(OUTPUT_FILE)}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    print("Next Steps:")
    print("  1. Send f1_complete_dataset.csv to preprocessing script")
    print("  2. Normalize weather variables")
    print("  3. Start model training")
    
    print("\nCache Statistics:")
    cache_info = requests_cache.get_cache()
    print(f"  - Cache file: {CACHE_DB}.sqlite")
    print(f"  - Cached requests: {len(cache_info.responses)}")


if __name__ == "__main__":
    try:
        main()
        print("\n* Process completed successfully")
    except KeyboardInterrupt:
        print("\n\n! Process interrupted by user")
    except Exception as e:
        print(f"\n\nX Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        requests_cache.uninstall_cache()
        print("\nCache closed.")
