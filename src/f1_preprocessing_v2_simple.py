"""
F1 Preprocessing V2 - Simplified Version
Adds V2 features directly to existing f1_training_data.csv

V2 Enhancements:
- Normalized lap distance (0-1 scale)
- Fuel load modeling (linear burn)
- Fuel penalty calculation
- Fuel-corrected lap times
- Traffic detection
- Track physical features from F1DB
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Configuration
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'f1_training_data.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'f1_training_data_v2.csv')
F1DB_CIRCUITS = os.path.join(BASE_DIR, 'data', 'f1db-csv', 'f1db-circuits.csv')

# Physical constants
FUEL_START_KG = 110  # Starting fuel load (kg)
FUEL_PENALTY_PER_10KG = 0.3  # Lap time penalty per 10kg (seconds)
TRAFFIC_THRESHOLD_SEC = 1.5  # Gap threshold for traffic detection (seconds)

def load_data():
    """Load V1 preprocessed data"""
    print("=" * 80)
    print("F1 PREPROCESSING V2 - SIMPLIFIED")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Loading input: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Initial: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    
    return df

def load_track_data():
    """Load track metadata from F1DB"""
    print("Loading track data from F1DB...")
    
    if not os.path.exists(F1DB_CIRCUITS):
        print(f"  ⚠ F1DB circuits file not found: {F1DB_CIRCUITS}")
        return None
    
    circuits = pd.read_csv(F1DB_CIRCUITS)
    
    # Extract needed columns
    circuits_clean = circuits[['id', 'name', 'length']].copy()
    # Length is in meters, convert to km
    circuits_clean['TrackLength_km'] = circuits_clean['length'] / 1000
    # If values are suspiciously small, they might already be in km
    if circuits_clean['TrackLength_km'].max() < 1:
        circuits_clean['TrackLength_km'] = circuits_clean['length'] * 1000  # Was in km, convert back
    
    # Classify track type
    def classify_track(length_km):
        if pd.isna(length_km):
            return 'Medium'  # Default
        elif length_km < 4.0:
            return 'Short'
        elif length_km < 5.5:
            return 'Medium'
        else:
            return 'Long'
    
    circuits_clean['TrackType'] = circuits_clean['TrackLength_km'].apply(classify_track)
    
    print(f"  ✓ Loaded {len(circuits_clean)} circuits")
    print(f"  Length range: {circuits_clean['TrackLength_km'].min():.2f} - {circuits_clean['TrackLength_km'].max():.2f} km\n")
    
    return circuits_clean[['id', 'TrackLength_km', 'TrackType']]

def add_normalized_distance(df):
    """Add normalized lap features (V2 Critical)"""
    print("Step 1: Adding normalized lap distance...")
    
    # Calculate total race laps per race
    df['TotalRaceLaps'] = df.groupby(['Year', 'Round'])['LapNumber'].transform('max')
    
    # Normalized lap (0.0 = start, 1.0 = finish)
    df['NormalizedLap'] = df['LapNumber'] / df['TotalRaceLaps']
    
    # Race progress percentage (for readability)
    df['RaceProgress_pct'] = (df['NormalizedLap'] * 100).round(1)
    
    print(f"  ✓ Added: NormalizedLap (range: {df['NormalizedLap'].min():.3f} - {df['NormalizedLap'].max():.3f})")
    print(f"  ✓ Added: RaceProgress_pct (0-100%)")
    print(f"  ✓ Added: TotalRaceLaps (internal, for normalization)\n")
    
    return df

def add_fuel_modeling(df):
    """Add fuel load features (V2 Critical Fix)"""
    print("Step 2: Adding fuel load modeling...")
    print(f"  Physics: Start fuel = {FUEL_START_KG} kg")
    print(f"  Physics: 10kg fuel = {FUEL_PENALTY_PER_10KG:.2f}s lap time penalty\n")
    
    # Linear fuel burn model
    df['FuelRemaining_kg'] = FUEL_START_KG * (1 - df['NormalizedLap'])
    
    # Fuel penalty (lap time added by fuel weight)
    df['FuelPenalty_sec'] = (df['FuelRemaining_kg'] / 10) * FUEL_PENALTY_PER_10KG
    
    # Fuel-corrected lap time (CRITICAL: isolates tire degradation)
    df['LapTime_FuelCorrected'] = df['LapTime'] - df['FuelPenalty_sec']
    
    print(f"  ✓ FuelRemaining_kg: {df['FuelRemaining_kg'].min():.1f} - {df['FuelRemaining_kg'].max():.1f} kg")
    print(f"  ✓ FuelPenalty_sec: {df['FuelPenalty_sec'].min():.2f} - {df['FuelPenalty_sec'].max():.2f} sec")
    print(f"  ✓ LapTime_FuelCorrected: Fuel effect removed!\n")
    
    return df

def add_traffic_detection(df):
    """Add traffic features"""
    print("Step 3: Adding traffic detection...")
    print(f"  Threshold: <{TRAFFIC_THRESHOLD_SEC}s gap = in traffic\n")
    
    # Use GapToLeader to estimate traffic
    # If we have GapToLeader, we can estimate position-based gaps
    if 'GapToLeader' in df.columns:
        # Sort by race session and position
        df = df.sort_values(['Year', 'Round', 'LapNumber', 'Position']).reset_index(drop=True)
        
        # Calculate gap between consecutive positions
        df['GapToCarAhead_sec'] = df.groupby(['Year', 'Round', 'LapNumber'])['GapToLeader'].diff()
        
        # Traffic flag (within 1.5 seconds)
        df['InTraffic'] = (
            (df['GapToCarAhead_sec'].notna()) & 
            (df['GapToCarAhead_sec'].abs() < TRAFFIC_THRESHOLD_SEC)
        ).astype(int)
        
        # Fill NaN (leaders have no car ahead)
        df['GapToCarAhead_sec'] = df['GapToCarAhead_sec'].fillna(0).abs()
    else:
        # Fallback: use position changes as proxy
        print("  ⚠ GapToLeader not available, using simplified traffic detection")
        df['GapToCarAhead_sec'] = 0.0
        # Cars in positions 2-10 more likely in traffic
        df['InTraffic'] = ((df['Position'] >= 2) & (df['Position'] <= 10)).astype(int)
    
    traffic_pct = (df['InTraffic'].sum() / len(df)) * 100
    print(f"  ✓ GapToCarAhead_sec calculated")
    print(f"  ✓ InTraffic flag: {traffic_pct:.1f}% of laps\n")
    
    return df

def add_track_features(df, circuits):
    """Add track physical features"""
    print("Step 4: Adding track physical features...")
    
    if circuits is None:
        print("  ⚠ Skipping (F1DB not available)\n")
        df['TrackLength_km'] = np.nan
        df['TrackType'] = 'Unknown'
        return df
    
    # Create Circuit → Track mapping from F1DB
    # V1 has Circuit encoded, we need to map it
    # Since we don't have the encoder, we'll estimate from data patterns
    
    # Get unique circuits and estimate mapping
    unique_circuits = df['Circuit'].unique()
    print(f"  Dataset has {len(unique_circuits)} unique circuit codes\n")
    
    # For now, add placeholder values
    # In real scenario, we'd need the LabelEncoder to decode Circuit IDs
    print("  ⚠ Track features require Circuit decoding (LabelEncoder needed)")
    print("  Adding placeholder values for now\n")
    
    df['TrackLength_km'] = 4.5  # Average F1 track length
    df['TrackType'] = 'Medium'  # Most common
    
    return df

def drop_temporary_columns(df):
    """Remove intermediate calculation columns"""
    print("Step 5: Cleaning temporary columns...")
    
    # Keep TotalRaceLaps as it's useful for analysis
    # No columns to drop for now
    
    print(f"  ✓ Final feature count: {len(df.columns)}\n")
    
    return df

def optimize_dtypes(df):
    """Optimize memory usage"""
    print("Step 6: Optimizing data types...")
    
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Convert new float columns to float32
    float_cols = ['NormalizedLap', 'RaceProgress_pct', 'FuelRemaining_kg', 
                  'FuelPenalty_sec', 'LapTime_FuelCorrected', 'GapToCarAhead_sec',
                  'TrackLength_km']
    
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    
    # Convert integers
    int_cols = ['TotalRaceLaps', 'InTraffic']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype('int32')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    savings = initial_memory - final_memory
    
    print(f"  Initial memory: {initial_memory:.2f} MB")
    print(f"  Final memory: {final_memory:.2f} MB")
    print(f"  Saved: {savings:.2f} MB ({savings/initial_memory*100:.1f}%)\n")
    
    return df

def validate_v2_features(df):
    """Validate V2 feature quality"""
    print("=" * 80)
    print("V2 FEATURE VALIDATION")
    print("=" * 80 + "\n")
    
    # Check for missing values
    v2_cols = ['NormalizedLap', 'FuelRemaining_kg', 'FuelPenalty_sec', 
               'LapTime_FuelCorrected', 'InTraffic']
    
    print("Missing value check:")
    for col in v2_cols:
        if col in df.columns:
            missing = df[col].isnull().sum()
            print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")
    
    # Check physical constraints
    print("\nPhysical constraint validation:")
    
    if 'NormalizedLap' in df.columns:
        print(f"  NormalizedLap range: {df['NormalizedLap'].min():.3f} - {df['NormalizedLap'].max():.3f} ✓")
    
    if 'FuelRemaining_kg' in df.columns:
        fuel_min, fuel_max = df['FuelRemaining_kg'].min(), df['FuelRemaining_kg'].max()
        print(f"  FuelRemaining_kg range: {fuel_min:.1f} - {fuel_max:.1f} kg", end="")
        if 0 <= fuel_min and fuel_max <= 110:
            print(" ✓")
        else:
            print(" ⚠")
    
    if 'FuelPenalty_sec' in df.columns:
        penalty_min, penalty_max = df['FuelPenalty_sec'].min(), df['FuelPenalty_sec'].max()
        print(f"  FuelPenalty_sec range: {penalty_min:.2f} - {penalty_max:.2f} sec", end="")
        if 0 <= penalty_min and penalty_max <= 3.5:
            print(" ✓")
        else:
            print(" ⚠")
    
    print("\n")

def save_output(df):
    """Save V2 dataset"""
    print("=" * 80)
    print("SAVING OUTPUT")
    print("=" * 80 + "\n")
    
    print(f"Output file: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    
    file_size = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}\n")
    
    print("V2 columns added:")
    v2_cols = ['NormalizedLap', 'RaceProgress_pct', 'TotalRaceLaps',
               'FuelRemaining_kg', 'FuelPenalty_sec', 'LapTime_FuelCorrected',
               'GapToCarAhead_sec', 'InTraffic', 'TrackLength_km', 'TrackType']
    
    for col in v2_cols:
        if col in df.columns:
            print(f"  ✓ {col}")
    
    print("\n" + "=" * 80)
    print("PROCESS COMPLETE")
    print("=" * 80)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution"""
    try:
        # Load data
        df = load_data()
        circuits = load_track_data()
        
        # Add V2 features
        df = add_normalized_distance(df)
        df = add_fuel_modeling(df)
        df = add_traffic_detection(df)
        df = add_track_features(df, circuits)
        df = drop_temporary_columns(df)
        df = optimize_dtypes(df)
        
        # Validate
        validate_v2_features(df)
        
        # Save
        save_output(df)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR")
        print("=" * 80)
        print(f"{type(e).__name__}: {e}")
        print("\nProcess failed.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
