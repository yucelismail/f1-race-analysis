# -*- coding: utf-8 -*-
"""
F1 Database Enrichment
Enriches the main F1 dataset using F1DB database files.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Configuration
MAIN_DATA_FILE = '../data/f1_race_analysis_data_enriched.csv'
OUTPUT_FILE = '../data/f1_ultimate_data.csv'
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


def ultimate_enrichment():
    """Main data enrichment pipeline."""
    print("="*80)
    print("F1 Database Enrichment")
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
    print("Data Merging")
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
    
    # 7. Circuit coordinates
    print("\n7. Circuit Coordinates (circuitId -> lat, lng)")
    circuits_df = f1db['circuits'][['id', 'name', 'latitude', 'longitude', 'placeName']].copy()
    circuits_df.columns = ['circuitId', 'CircuitName', 'CircuitLat', 'CircuitLng', 'CircuitLocation']
    
    df = df.merge(circuits_df, on='circuitId', how='left')
    matched = df['CircuitLat'].notna().sum()
    print(f"  * {matched:,} / {len(df):,} rows matched ({(matched/len(df))*100:.1f}%)")
    
    # 8. Cleanup
    print("\n8. Cleanup - Removing temporary columns")
    columns_to_drop = [
        'raceId', 'driverId', 'constructorId', 'circuitId',
        'DriverCode', 'DriverDOB', 'RaceDate'
    ]
    
    existing_drops = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"  * {len(existing_drops)} columns removed: {', '.join(existing_drops)}")
    
    # 9. Save
    print("\n" + "="*80)
    print("Saving")
    print("="*80)
    
    df.to_csv(OUTPUT_FILE, index=False)
    file_size = os.path.getsize(OUTPUT_FILE) / (1024*1024)
    print(f"  * File saved: {OUTPUT_FILE}")
    print(f"  * File size: {file_size:.2f} MB")
    
    # 10. Summary
    print("\n" + "="*80)
    print("Enrichment Summary")
    print("="*80)
    print(f"  - Initial rows       : {initial_rows:,}")
    print(f"  - Final rows         : {len(df):,}")
    print(f"  - Row change         : {len(df) - initial_rows:+,}")
    print(f"  - Initial columns    : {initial_cols}")
    print(f"  - Final columns      : {len(df.columns)}")
    print(f"  - Added columns      : {len(df.columns) - initial_cols + len(existing_drops)}")
    print(f"  - Removed columns    : {len(existing_drops)}")
    
    new_columns = [
        'DriverFullName', 'DriverAge', 
        'DriverStandingsPoints', 'DriverStandingsPosition',
        'ConstructorStandingsPoints', 'ConstructorStandingsPosition',
        'CircuitName', 'CircuitLat', 'CircuitLng', 'CircuitLocation'
    ]
    
    print(f"\nAdded columns:")
    for col in new_columns:
        if col in df.columns:
            non_null = df[col].notna().sum()
            coverage = (non_null / len(df)) * 100
            sample = df[col].dropna().iloc[0] if non_null > 0 else 'N/A'
            print(f"  - {col:<30} : {non_null:>7,} data ({coverage:>5.1f}%) | Sample: {sample}")
    
    print("\n" + "="*80)
    print("Data Preview (First 5 Rows)")
    print("="*80 + "\n")
    
    preview_cols = [
        'Year', 'Round', 'Driver', 'DriverFullName', 'DriverAge',
        'LapNumber', 'LapTime', 'Position',
        'DriverStandingsPoints', 'DriverStandingsPosition',
        'ConstructorStandingsPoints', 'ConstructorStandingsPosition',
        'CircuitName', 'CircuitLat', 'CircuitLng', 'CircuitLocation'
    ]
    
    available_preview = [col for col in preview_cols if col in df.columns]
    
    with pd.option_context('display.max_columns', None, 'display.width', 150):
        print(df[available_preview].head(5).to_string(index=False))
    
    print("\n" + "="*80)
    print("Column Data Types")
    print("="*80)
    
    for col in new_columns:
        if col in df.columns:
            dtype = df[col].dtype
            unique = df[col].nunique()
            print(f"  - {col:<30} : {str(dtype):<10} ({unique:>5,} unique values)")
    
    print("\n" + "="*80)
    print("Database Enrichment Complete")
    print(f"File: {os.path.abspath(OUTPUT_FILE)}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    print("Next Steps:")
    print("  1. Send enriched data to preprocessing script")
    print("  2. Encode categorical columns (CircuitName, etc.)")
    print("  3. Normalize continuous variables (age, points, coordinates)")
    print("  4. Start model training")
    
    return df


if __name__ == "__main__":
    enriched_df = ultimate_enrichment()
    
    if enriched_df is not None:
        print(f"\nProcess completed successfully!")
        print(f"Enriched dataset: {len(enriched_df):,} rows x {len(enriched_df.columns)} columns")
    else:
        print("\nProcess failed. Check error messages above.")
