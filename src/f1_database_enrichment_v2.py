# -*- coding: utf-8 -*-
"""
F1 Database Enrichment V2
Enhanced version with track length and physical constraints.

V2 Changes:
- Added TrackLength from F1DB circuits
- Added track type classification
- Improved circuit metadata extraction
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Configuration
MAIN_DATA_FILE = '../data/f1_training_data.csv'  # V2: Use existing V1 output
OUTPUT_FILE = '../data/f1_ultimate_data_v2.csv'
F1DB_DIR = '../data/f1db-csv'

F1DB_FILES = {
    'races': 'f1db-races.csv',
    'drivers': 'f1db-drivers.csv',
    'results': 'f1db-races-race-results.csv',
    'driver_standings': 'f1db-races-driver-standings.csv',
    'constructor_standings': 'f1db-races-constructor-standings.csv',
    'circuits': 'f1db-circuits.csv'
}

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def check_files_exist():
    """Check if required files exist."""
    print("\nFile checks...")
    
    missing_files = []
    
    if not os.path.exists(MAIN_DATA_FILE):
        missing_files.append(MAIN_DATA_FILE)
        print(f"  X {MAIN_DATA_FILE} not found!")
    else:
        size = os.path.getsize(MAIN_DATA_FILE) / (1024*1024)
        print(f"  * {MAIN_DATA_FILE} ({size:.2f} MB)")
    
    if not os.path.exists(F1DB_DIR):
        print(f"\n  X {F1DB_DIR}/ directory not found!")
        missing_files.append(F1DB_DIR)
    else:
        print(f"  * {F1DB_DIR}/ directory found")
        
        for key, filename in F1DB_FILES.items():
            filepath = os.path.join(F1DB_DIR, filename)
            if not os.path.exists(filepath):
                missing_files.append(filepath)
                print(f"  X {filepath} not found!")
            else:
                size = os.path.getsize(filepath) / 1024
                print(f"  * {filepath} ({size:.2f} KB)")
    
    if missing_files:
        print(f"\nERROR: {len(missing_files)} files missing!")
        print("Missing files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nDownload F1DB files from: https://github.com/f1db/f1db/releases")
        return False
    
    print("  All files present!\n")
    return True


def load_f1db_files():
    """Load F1DB CSV files from directory."""
    print(f"Loading F1DB files from {F1DB_DIR}/...")
    
    f1db_data = {}
    
    for key, filename in F1DB_FILES.items():
        try:
            filepath = os.path.join(F1DB_DIR, filename)
            df = pd.read_csv(filepath)
            f1db_data[key] = df
            print(f"  * {filepath:<40} : {len(df):>6,} rows")
        except Exception as e:
            print(f"  X {filepath} failed to load: {e}")
            return None
    
    return f1db_data


def calculate_age(birth_date, race_date):
    """Calculate age between two dates (decimal years)."""
    try:
        if pd.isna(birth_date) or pd.isna(race_date):
            return np.nan
        
        birth = pd.to_datetime(birth_date)
        race = pd.to_datetime(race_date)
        
        age_years = race.year - birth.year
        age_days = (race - birth.replace(year=race.year)).days
        
        if age_days < 0:
            age_years -= 1
            age_days = (race - birth.replace(year=race.year - 1)).days
        
        age = age_years + (age_days / 365.25)
        return round(age, 2)
    except:
        return np.nan


def classify_track_type(length_km):
    """
    Classify track by length into categories.
    V2 Feature: Track type classification for better modeling.
    """
    if pd.isna(length_km):
        return 'Unknown'
    elif length_km < 4.0:
        return 'Short'  # Monaco, Zandvoort
    elif length_km < 5.5:
        return 'Medium'  # Most tracks
    else:
        return 'Long'  # Spa, Silverstone


def ultimate_enrichment():
    """
    Main data enrichment pipeline V2.
    
    V2 Enhancements:
    - Track length extraction
    - Track type classification
    - Total race laps calculation
    """
    print("="*80)
    print("F1 Database Enrichment V2")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not check_files_exist():
        return None
    
    print(f"Loading main data: {MAIN_DATA_FILE}")
    df = pd.read_csv(MAIN_DATA_FILE)
    print(f"  * {len(df):,} rows, {len(df.columns)} columns\n")
    
    initial_rows = len(df)
    initial_cols = len(df.columns)
    
    f1db = load_f1db_files()
    if f1db is None:
        return None
    
    print("\n" + "="*80)
    print("Data Merging - V2 Enhanced")
    print("="*80 + "\n")
    
    # 1. Race ID matching
    print("1. Race ID Matching (Year, Round -> raceId, date, circuitId)")
    races_df = f1db['races'][['id', 'year', 'round', 'circuitId', 'date']].copy()
    races_df.columns = ['raceId', 'Year', 'Round', 'circuitId', 'RaceDate']
    
    df['Year'] = df['Year'].astype(int)
    df['Round'] = df['Round'].astype(int)
    races_df['Year'] = races_df['Year'].astype(int)
    races_df['Round'] = races_df['Round'].astype(int)
    
    df = df.merge(races_df, on=['Year', 'Round'], how='left')
    matched = df['raceId'].notna().sum()
    print(f"  * {matched:,} / {len(df):,} rows matched ({(matched/len(df))*100:.1f}%)")
    
    # 2. Driver ID matching
    print("\n2. Driver ID Matching (Driver -> driverId, dob)")
    drivers_df = f1db['drivers'][['id', 'abbreviation', 'dateOfBirth', 'fullName']].copy()
    drivers_df.columns = ['driverId', 'DriverCode', 'DriverDOB', 'DriverFullName']
    
    df['Driver'] = df['Driver'].astype(str)
    drivers_df['DriverCode'] = drivers_df['DriverCode'].astype(str)
    
    df = df.merge(drivers_df, left_on='Driver', right_on='DriverCode', how='left')
    matched = df['driverId'].notna().sum()
    print(f"  * {matched:,} / {len(df):,} rows matched ({(matched/len(df))*100:.1f}%)")
    
    # 3. Driver age calculation
    print("\n3. Driver Age Calculation (RaceDate - DOB)")
    df['DriverAge'] = df.apply(
        lambda row: calculate_age(row['DriverDOB'], row['RaceDate']), 
        axis=1
    )
    calculated = df['DriverAge'].notna().sum()
    print(f"  * {calculated:,} ages calculated")
    if calculated > 0:
        print(f"     Mean age: {df['DriverAge'].mean():.1f}")
        print(f"     Min: {df['DriverAge'].min():.1f}, Max: {df['DriverAge'].max():.1f}")
    
    # 4. Constructor ID matching
    print("\n4. Constructor ID Matching (raceId, driverId -> constructorId)")
    results_df = f1db['results'][['raceId', 'driverId', 'constructorId']].copy()
    results_df = results_df.drop_duplicates(subset=['raceId', 'driverId'])
    
    df = df.merge(results_df, on=['raceId', 'driverId'], how='left')
    matched = df['constructorId'].notna().sum()
    print(f"  * {matched:,} / {len(df):,} rows matched ({(matched/len(df))*100:.1f}%)")
    
    # 5. Driver standings
    print("\n5. Driver Championship Points (raceId, driverId -> points, position)")
    driver_standings = f1db['driver_standings'][['raceId', 'driverId', 'points', 'positionNumber']].copy()
    driver_standings.columns = ['raceId', 'driverId', 'DriverStandingsPoints', 'DriverStandingsPosition']
    
    df = df.merge(driver_standings, on=['raceId', 'driverId'], how='left')
    
    df['DriverStandingsPoints'] = df['DriverStandingsPoints'].fillna(0)
    df['DriverStandingsPosition'] = df['DriverStandingsPosition'].fillna(0)
    
    matched = (df['DriverStandingsPoints'] > 0).sum()
    print(f"  * {matched:,} rows with points data")
    print(f"     Missing values filled with 0")
    
    # 6. Constructor standings
    print("\n6. Constructor Championship Points (raceId, constructorId -> points, position)")
    constructor_standings = f1db['constructor_standings'][['raceId', 'constructorId', 'points', 'positionNumber']].copy()
    constructor_standings.columns = ['raceId', 'constructorId', 'ConstructorStandingsPoints', 'ConstructorStandingsPosition']
    
    df = df.merge(constructor_standings, on=['raceId', 'constructorId'], how='left')
    
    df['ConstructorStandingsPoints'] = df['ConstructorStandingsPoints'].fillna(0)
    df['ConstructorStandingsPosition'] = df['ConstructorStandingsPosition'].fillna(0)
    
    matched = (df['ConstructorStandingsPoints'] > 0).sum()
    print(f"  * {matched:,} rows with constructor points")
    print(f"     Missing values filled with 0")
    
    # 7. Circuit coordinates and V2: TRACK LENGTH
    print("\n7. [V2 ENHANCED] Circuit Data (circuitId -> lat, lng, length, type)")
    circuits_df = f1db['circuits'][['id', 'name', 'latitude', 'longitude', 'placeName', 'length']].copy()
    circuits_df.columns = ['circuitId', 'CircuitName', 'CircuitLat', 'CircuitLng', 'CircuitLocation', 'TrackLength_m']
    
    # V2: Convert length to kilometers
    circuits_df['TrackLength_km'] = circuits_df['TrackLength_m'] / 1000
    
    # V2: Classify track type
    circuits_df['TrackType'] = circuits_df['TrackLength_km'].apply(classify_track_type)
    
    df = df.merge(circuits_df, on='circuitId', how='left')
    matched = df['CircuitLat'].notna().sum()
    print(f"  * {matched:,} / {len(df):,} rows matched ({(matched/len(df))*100:.1f}%)")
    
    # V2: Show track length statistics
    if 'TrackLength_km' in df.columns:
        valid_lengths = df['TrackLength_km'].dropna()
        if len(valid_lengths) > 0:
            print(f"  * [V2] Track length range: {valid_lengths.min():.3f} - {valid_lengths.max():.3f} km")
            print(f"  * [V2] Track types: {df['TrackType'].value_counts().to_dict()}")
    
    # 8. V2: Total Race Laps Calculation
    print("\n8. [V2 NEW] Total Race Laps Calculation")
    df['TotalRaceLaps'] = df.groupby(['Year', 'Round'])['LapNumber'].transform('max')
    print(f"  * Total race laps added for normalization")
    print(f"     Range: {df['TotalRaceLaps'].min():.0f} - {df['TotalRaceLaps'].max():.0f} laps")
    
    # 9. Cleanup
    print("\n9. Cleanup - Removing temporary columns")
    columns_to_drop = [
        'raceId', 'driverId', 'constructorId', 'circuitId',
        'DriverCode', 'DriverDOB', 'RaceDate', 'TrackLength_m'
    ]
    
    existing_drops = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"  * {len(existing_drops)} columns removed: {', '.join(existing_drops)}")
    
    # 10. Save
    print("\n" + "="*80)
    print("Saving V2")
    print("="*80)
    
    df.to_csv(OUTPUT_FILE, index=False)
    file_size = os.path.getsize(OUTPUT_FILE) / (1024*1024)
    print(f"  * File saved: {OUTPUT_FILE}")
    print(f"  * File size: {file_size:.2f} MB")
    
    # 11. Summary
    print("\n" + "="*80)
    print("V2 Enrichment Summary")
    print("="*80)
    print(f"  - Initial rows       : {initial_rows:,}")
    print(f"  - Final rows         : {len(df):,}")
    print(f"  - Row change         : {len(df) - initial_rows:+,}")
    print(f"  - Initial columns    : {initial_cols}")
    print(f"  - Final columns      : {len(df.columns)}")
    print(f"  - Added columns      : {len(df.columns) - initial_cols + len(existing_drops)}")
    print(f"  - Removed columns    : {len(existing_drops)}")
    
    new_columns_v1 = [
        'DriverFullName', 'DriverAge', 
        'DriverStandingsPoints', 'DriverStandingsPosition',
        'ConstructorStandingsPoints', 'ConstructorStandingsPosition',
        'CircuitName', 'CircuitLat', 'CircuitLng', 'CircuitLocation'
    ]
    
    new_columns_v2 = [
        'TrackLength_km', 'TrackType', 'TotalRaceLaps'
    ]
    
    print(f"\n[V1] Original columns:")
    for col in new_columns_v1:
        if col in df.columns:
            non_null = df[col].notna().sum()
            coverage = (non_null / len(df)) * 100
            print(f"  - {col:<30} : {non_null:>7,} data ({coverage:>5.1f}%)")
    
    print(f"\n[V2] NEW columns:")
    for col in new_columns_v2:
        if col in df.columns:
            non_null = df[col].notna().sum()
            coverage = (non_null / len(df)) * 100
            sample = df[col].dropna().iloc[0] if non_null > 0 else 'N/A'
            print(f"  - {col:<30} : {non_null:>7,} data ({coverage:>5.1f}%) | Sample: {sample}")
    
    print("\n" + "="*80)
    print("V2 Database Enrichment Complete")
    print(f"File: {os.path.abspath(OUTPUT_FILE)}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    print("V2 Enhancements:")
    print("  1. Track length (km) from F1DB circuits")
    print("  2. Track type classification (Short/Medium/Long)")
    print("  3. Total race laps for normalization")
    print("\nNext: Run f1_preprocessing_v2.py for fuel load modeling")
    
    return df


if __name__ == "__main__":
    enriched_df = ultimate_enrichment()
    
    if enriched_df is not None:
        print(f"\nV2 Process completed successfully!")
        print(f"Enriched dataset: {len(enriched_df):,} rows x {len(enriched_df.columns)} columns")
    else:
        print("\nProcess failed. Check error messages above.")
