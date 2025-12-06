# -*- coding: utf-8 -*-
"""
F1 Machine Learning Preprocessing V2
Enhanced version with fuel load modeling and normalization.

V2 Critical Changes:
- Normalized lap distance (0-1)
- Fuel remaining estimation
- Fuel penalty calculation (10kg = 0.3s rule)
- Gap to car ahead (traffic modeling)
- Fuel-corrected lap times
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Configuration
INPUT_FILE = '../data/f1_ultimate_data_v2.csv'
OUTPUT_FILE = '../data/f1_training_data_v2.csv'
ENCODERS_FILE = '../data/f1_encoders_v2.pkl'

# V2: Physical constants
FUEL_START_KG = 110  # Average race start fuel load (kg)
FUEL_PENALTY_PER_10KG = 0.3  # 10kg fuel = 0.3 seconds lap time penalty
TRAFFIC_THRESHOLD_SEC = 1.5  # Gap threshold for traffic effect

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("="*80)
print("F1 Machine Learning Preprocessing V2")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# Step 1: Load Data
print("Step 1: Loading data...")

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found!")
    print("Run f1_database_enrichment_v2.py first!")
    exit(1)

df = pd.read_csv(INPUT_FILE)
initial_rows = len(df)
initial_cols = len(df.columns)

print(f"  * Loaded: {len(df):,} rows x {len(df.columns)} columns")
print(f"  * File size: {os.path.getsize(INPUT_FILE) / (1024*1024):.2f} MB\n")


# Step 2: V2 NEW - Normalized Race Distance
print("="*80)
print("Step 2: [V2 NEW] Normalized Race Distance Calculation")
print("="*80)

# Ensure TotalRaceLaps exists
if 'TotalRaceLaps' not in df.columns:
    print("  ! TotalRaceLaps not found, calculating...")
    df['TotalRaceLaps'] = df.groupby(['Year', 'Round'])['LapNumber'].transform('max')

# Calculate normalized lap (0.0 = race start, 1.0 = race end)
df['NormalizedLap'] = df['LapNumber'] / df['TotalRaceLaps']

print(f"  * NormalizedLap calculated")
print(f"     Range: {df['NormalizedLap'].min():.3f} - {df['NormalizedLap'].max():.3f}")
print(f"     Mean: {df['NormalizedLap'].mean():.3f}")

# Race progress percentage
df['RaceProgress_pct'] = (df['NormalizedLap'] * 100).round(1)

print(f"  * RaceProgress_pct added (0-100%)")
print()


# Step 3: V2 NEW - Fuel Load Modeling
print("="*80)
print("Step 3: [V2 NEW] Fuel Load and Penalty Calculation")
print("="*80)

# Assume linear fuel burn (simplified model)
df['FuelRemaining_kg'] = FUEL_START_KG * (1 - df['NormalizedLap'])

# Calculate fuel penalty (industry standard: 10kg = 0.3s)
df['FuelPenalty_sec'] = (df['FuelRemaining_kg'] / 10) * FUEL_PENALTY_PER_10KG

print(f"  * FuelRemaining_kg calculated")
print(f"     Start fuel: {FUEL_START_KG} kg")
print(f"     Range: {df['FuelRemaining_kg'].min():.1f} - {df['FuelRemaining_kg'].max():.1f} kg")

print(f"  * FuelPenalty_sec calculated (10kg = {FUEL_PENALTY_PER_10KG}s)")
print(f"     Range: {df['FuelPenalty_sec'].min():.2f} - {df['FuelPenalty_sec'].max():.2f} seconds")
print(f"     Mean penalty: {df['FuelPenalty_sec'].mean():.2f}s")

# V2: Fuel-corrected lap time (removes fuel effect)
if 'LapTime' in df.columns:
    df['LapTime_FuelCorrected'] = df['LapTime'] - df['FuelPenalty_sec']
    print(f"  * LapTime_FuelCorrected added (LapTime - FuelPenalty)")
    print(f"     This isolates tire degradation effect!")

print()


# Step 4: V2 NEW - Traffic Modeling (Gap to Car Ahead)
print("="*80)
print("Step 4: [V2 NEW] Traffic and Gap Analysis")
print("="*80)

# Sort by race session and position
df = df.sort_values(['Year', 'Round', 'LapNumber', 'Position']).reset_index(drop=True)

# Calculate cumulative time if not present
if 'CumulativeTime' not in df.columns and 'LapTime' in df.columns:
    print("  ! CumulativeTime not found, calculating from LapTime...")
    df['CumulativeTime'] = df.groupby(['Year', 'Round', 'Driver'])['LapTime'].cumsum()

# Gap to car ahead (time difference to car in front)
if 'CumulativeTime' in df.columns:
    df['GapToCarAhead_sec'] = df.groupby(['Year', 'Round', 'LapNumber'])['CumulativeTime'].diff()
    
    # First position has no gap
    df['GapToCarAhead_sec'] = df['GapToCarAhead_sec'].fillna(0)
    
    # Traffic flag: within 1.5 seconds of car ahead
    df['InTraffic'] = (df['GapToCarAhead_sec'] > 0) & (df['GapToCarAhead_sec'] < TRAFFIC_THRESHOLD_SEC)
    df['InTraffic'] = df['InTraffic'].astype(int)
    
    traffic_count = df['InTraffic'].sum()
    traffic_pct = (traffic_count / len(df)) * 100
    
    print(f"  * GapToCarAhead_sec calculated")
    print(f"     Range: {df['GapToCarAhead_sec'].min():.2f} - {df['GapToCarAhead_sec'].max():.2f} seconds")
    print(f"  * InTraffic flag added (<{TRAFFIC_THRESHOLD_SEC}s gap)")
    print(f"     Laps in traffic: {traffic_count:,} ({traffic_pct:.1f}%)")
else:
    print("  ! CumulativeTime not available, skipping traffic analysis")

print()


# Step 5: V2 ENHANCED - Remove Unnecessary Columns
print("="*80)
print("Step 5: [V2 ENHANCED] Removing unnecessary columns...")
print("="*80)

columns_to_drop = [
    # V1: Original drops
    'DriverName', 'DriverAbbr', 'DriverFullName',
    'CircuitName', 'CircuitLocation',
    'Country', 'Location',
    'PitOutTime', 'PitInTime',
    'DriverNumber',
    
    # V2: Additional metadata
    'TotalRaceLaps',  # Already encoded in NormalizedLap
]

existing_drops = [col for col in columns_to_drop if col in df.columns]

if existing_drops:
    df = df.drop(columns=existing_drops)
    print(f"  * {len(existing_drops)} columns removed")
    print(f"  * Remaining columns: {len(df.columns)}\n")
else:
    print("  * No columns to remove\n")


# Step 6: Missing Data Imputation
print("="*80)
print("Step 6: Filling missing data...")
print("="*80)

# SpeedST - Linear Interpolation
if 'SpeedST' in df.columns:
    missing_speed = df['SpeedST'].isna().sum()
    if missing_speed > 0:
        print(f"  - SpeedST: {missing_speed:,} missing values")
        
        df = df.sort_values(['Year', 'Round', 'Driver', 'LapNumber'])
        
        df['SpeedST'] = df.groupby(['Year', 'Round', 'Driver'])['SpeedST'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        
        df['SpeedST'] = df.groupby(['Year', 'Round', 'Driver'])['SpeedST'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
        
        remaining = df['SpeedST'].isna().sum()
        if remaining > 0:
            mean_speed = df['SpeedST'].mean()
            df['SpeedST'].fillna(mean_speed, inplace=True)
            print(f"    * Filled with interpolation + mean ({remaining} values)")
        else:
            print(f"    * Filled with interpolation")

# GapToLeader - Fill with 0.0
if 'GapToLeader' in df.columns:
    missing_gap = df['GapToLeader'].isna().sum()
    if missing_gap > 0:
        print(f"  - GapToLeader: {missing_gap:,} missing values")
        df['GapToLeader'].fillna(0.0, inplace=True)
        print(f"    * Filled with 0.0")

# Status - Fill with 'Unknown'
if 'Status' in df.columns:
    missing_status = df['Status'].isna().sum()
    if missing_status > 0:
        print(f"  - Status: {missing_status:,} missing values")
        df['Status'].fillna('Unknown', inplace=True)
        print(f"    * Filled with 'Unknown'")

# Other numeric columns - Fill with 0
numeric_cols_to_fill = ['Points', 'ClassifiedPosition', 'DriverStandingsPoints', 
                        'DriverStandingsPosition', 'ConstructorStandingsPoints', 
                        'ConstructorStandingsPosition']

for col in numeric_cols_to_fill:
    if col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  - {col}: {missing:,} missing values")
            df[col].fillna(0, inplace=True)
            print(f"    * Filled with 0")

print()


# Step 7: Categorical Encoding (Label Encoding)
print("="*80)
print("Step 7: Encoding categorical data...")
print("="*80)

categorical_columns = ['Driver', 'Team', 'Circuit', 'Compound', 'Status', 'TrackType']
encoders = {}

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].fillna('UNKNOWN')
        
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
        unique_count = len(le.classes_)
        print(f"  * {col:<15} : {unique_count:>3} unique values -> [0-{unique_count-1}] IDs")
    else:
        print(f"  ! {col} column not found, skipping...")

print(f"  * Total {len(encoders)} encoders saved")
print(f"  * [V2] Added: TrackType encoding\n")


# Step 8: Boolean Conversion
print("="*80)
print("Step 8: Converting boolean columns...")
print("="*80)

boolean_columns = ['FreshTyre', 'IsAccurate', 'Rainfall', 'InTraffic']

for col in boolean_columns:
    if col in df.columns:
        original_type = df[col].dtype
        
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
            print(f"  * {col:<15} : {original_type} -> int (0/1)")
        elif df[col].dtype == 'object':
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
            df[col] = df[col].fillna(0).astype(int)
            print(f"  * {col:<15} : object -> int (0/1)")
        else:
            print(f"  * {col:<15} : Already numeric ({original_type})")

print(f"  * [V2] Added: InTraffic flag\n")


# Step 9: Data Type Optimization
print("="*80)
print("Step 9: Optimizing data types...")
print("="*80)

# Float64 -> Float32
float_cols = df.select_dtypes(include=['float64']).columns
for col in float_cols:
    df[col] = df[col].astype('float32')

print(f"  * {len(float_cols)} float64 columns -> float32")

# Int64 -> Int32
int_cols = df.select_dtypes(include=['int64']).columns
for col in int_cols:
    if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
        df[col] = df[col].astype('int32')

print(f"  * {len(int_cols)} int64 columns -> int32\n")


# Step 10: Final Checks
print("="*80)
print("Step 10: Final checks...")
print("="*80)

# Check for non-numeric columns
non_numeric = df.select_dtypes(include=['object']).columns.tolist()
if non_numeric:
    print(f"  ! {len(non_numeric)} non-numeric columns found:")
    for col in non_numeric:
        print(f"     - {col} ({df[col].dtype})")
    print(f"     Removing these columns...")
    df = df.drop(columns=non_numeric)
else:
    print(f"  * All columns are numeric ({len(df.columns)} columns)")

# Check for NaN values
nan_count = df.isna().sum().sum()
if nan_count > 0:
    print(f"  ! {nan_count} NaN values found")
    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    for col, count in nan_cols.items():
        print(f"     - {col}: {count:,} NaN")
    
    print(f"     Removing rows with NaN...")
    df = df.dropna()
    print(f"     * {len(df):,} clean rows remaining")
else:
    print(f"  * No NaN values")

# Check for infinite values
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
if inf_count > 0:
    print(f"  ! {inf_count} infinite values found")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"     * Infinite values cleaned")
else:
    print(f"  * No infinite values")

print()


# Step 11: Save
print("="*80)
print("Step 11: Saving files...")
print("="*80)

# Save processed data
df.to_csv(OUTPUT_FILE, index=False)
file_size = os.path.getsize(OUTPUT_FILE) / (1024*1024)
print(f"  * Data saved: {OUTPUT_FILE} ({file_size:.2f} MB)")

# Save encoders
if encoders:
    joblib.dump(encoders, ENCODERS_FILE)
    encoder_size = os.path.getsize(ENCODERS_FILE) / 1024
    print(f"  * Encoders saved: {ENCODERS_FILE} ({encoder_size:.2f} KB)")
    print(f"     Content: {list(encoders.keys())}")

print()


# Step 12: V2 Summary Statistics
print("="*80)
print("V2 Preprocessing Summary")
print("="*80)
print(f"  - Initial rows       : {initial_rows:,}")
print(f"  - Final rows         : {len(df):,}")
print(f"  - Removed rows       : {initial_rows - len(df):,} ({((initial_rows - len(df))/initial_rows)*100:.2f}%)")
print(f"  - Initial columns    : {initial_cols}")
print(f"  - Final columns      : {len(df.columns)}")
print(f"  - Added columns (V2) : +6 (NormalizedLap, FuelRemaining, FuelPenalty, etc.)")
print(f"  - Total encoders     : {len(encoders)}")
print(f"  - Memory usage       : {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
print()

# V2 Specific Stats
print("="*80)
print("[V2] NEW FEATURES STATISTICS")
print("="*80)

v2_features = [
    'NormalizedLap', 'RaceProgress_pct', 'FuelRemaining_kg', 
    'FuelPenalty_sec', 'LapTime_FuelCorrected', 'GapToCarAhead_sec', 
    'InTraffic', 'TrackLength_km', 'TrackType'
]

for feat in v2_features:
    if feat in df.columns:
        non_null = df[feat].notna().sum()
        coverage = (non_null / len(df)) * 100
        
        if df[feat].dtype in ['float32', 'float64', 'int32', 'int64']:
            mean_val = df[feat].mean()
            std_val = df[feat].std()
            print(f"  - {feat:<25} : {non_null:>7,} ({coverage:>5.1f}%) | Mean: {mean_val:>8.2f} ± {std_val:.2f}")
        else:
            unique = df[feat].nunique()
            print(f"  - {feat:<25} : {non_null:>7,} ({coverage:>5.1f}%) | {unique} unique values")

print("\n" + "="*80)
print("V2 Preprocessing Complete")
print(f"Training data: {os.path.abspath(OUTPUT_FILE)}")
print(f"Encoder file: {os.path.abspath(ENCODERS_FILE)}")
print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

print("="*80)
print("[V2] KEY IMPROVEMENTS")
print("="*80)
print("1. Normalized lap distance (0-1) for cross-track comparison")
print("2. Fuel load estimation (linear burn model)")
print("3. Fuel penalty calculation (10kg = 0.3s industry standard)")
print("4. Fuel-corrected lap times (isolates tire degradation)")
print("5. Traffic detection (gap < 1.5s to car ahead)")
print("6. Track length and type classification")
print()
print("Next Steps:")
print("  1. Run f1_data_analysis_v2.py to compare correlations")
print("  2. Check if TyreLife correlation is now POSITIVE")
print("  3. Train models with fuel-corrected features")
print("="*80 + "\n")

print(f"V2 Process completed successfully!")
print(f"{len(df):,} rows x {len(df.columns)} columns ready for training\n")
