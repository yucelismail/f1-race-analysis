"""
F1 Data Cleaning V3 - Outlier Removal and Quality Filtering

V3 Critical Goal: Clean the dataset to fix negative TyreLife correlation

Root Cause Analysis:
- V1/V2 had TyreLife ↔ LapTime = -0.20 (physically impossible)
- Cause: Pit stops, safety cars, yellow flags contaminating data
- Solution: Remove all non-representative laps

V3 Cleaning Strategy:
1. Remove pit stop laps (LapTime > 120s)
2. Remove outlier laps (LapTime < 60s or > 120s)
3. Filter by track status (only green flag laps)
4. Remove inaccurate timing data
5. Remove first/last laps of race
6. Remove first lap of each stint (cold tires)
7. Validate data quality after cleaning

Expected Outcome:
- TyreLife ↔ LapTime_FuelCorrected should become POSITIVE (+0.4 to +0.6)
- Cleaner signal for ML modeling
- Better fuel effect correlation
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'f1_training_data_v2.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'f1_training_data_v3.csv')

# Cleaning thresholds
LAP_TIME_MIN = 60.0  # Minimum valid lap time (seconds)
LAP_TIME_MAX = 120.0  # Maximum valid lap time (seconds) - excludes pit stops
TRACK_STATUS_GREEN = 1  # Only green flag laps (1=Green in encoded data)
MIN_STINT_LAP = 2  # Skip first lap of stint (cold tires)

def load_data():
    """Load V2 dataset"""
    print("=" * 80)
    print("F1 DATA CLEANING V3")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Loading V2 data: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"  Initial: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    
    return df

def analyze_initial_quality(df):
    """Analyze data quality before cleaning"""
    print("=" * 80)
    print("INITIAL DATA QUALITY ANALYSIS")
    print("=" * 80 + "\n")
    
    print("Lap Time Distribution:")
    print(f"  Min: {df['LapTime'].min():.2f}s")
    print(f"  Max: {df['LapTime'].max():.2f}s")
    print(f"  Mean: {df['LapTime'].mean():.2f}s")
    print(f"  Median: {df['LapTime'].median():.2f}s")
    print(f"  Std: {df['LapTime'].std():.2f}s\n")
    
    # Identify problematic laps
    pit_laps = (df['LapTime'] > LAP_TIME_MAX).sum()
    too_fast = (df['LapTime'] < LAP_TIME_MIN).sum()
    
    print("Problematic Laps:")
    print(f"  Too slow (>120s, likely pit stops): {pit_laps:,} ({pit_laps/len(df)*100:.2f}%)")
    print(f"  Too fast (<60s, incomplete laps): {too_fast:,} ({too_fast/len(df)*100:.2f}%)")
    
    if 'TrackStatus' in df.columns:
        non_green = (df['TrackStatus'] != TRACK_STATUS_GREEN).sum()
        print(f"  Non-green flag laps: {non_green:,} ({non_green/len(df)*100:.2f}%)")
    
    if 'IsAccurate' in df.columns:
        inaccurate = (df['IsAccurate'] == 0).sum()
        print(f"  Inaccurate timing: {inaccurate:,} ({inaccurate/len(df)*100:.2f}%)")
    
    print()

def remove_outlier_lap_times(df):
    """Remove laps with outlier lap times"""
    print("=" * 80)
    print("STEP 1: REMOVING OUTLIER LAP TIMES")
    print("=" * 80 + "\n")
    
    initial_count = len(df)
    
    print(f"Thresholds:")
    print(f"  Minimum valid lap time: {LAP_TIME_MIN}s")
    print(f"  Maximum valid lap time: {LAP_TIME_MAX}s\n")
    
    # Filter lap times
    df_clean = df[(df['LapTime'] >= LAP_TIME_MIN) & (df['LapTime'] <= LAP_TIME_MAX)].copy()
    
    removed = initial_count - len(df_clean)
    print(f"Results:")
    print(f"  Removed: {removed:,} laps ({removed/initial_count*100:.2f}%)")
    print(f"  Remaining: {len(df_clean):,} laps ({len(df_clean)/initial_count*100:.2f}%)\n")
    
    return df_clean

def filter_track_status(df):
    """Keep only green flag laps"""
    print("=" * 80)
    print("STEP 2: FILTERING TRACK STATUS")
    print("=" * 80 + "\n")
    
    if 'TrackStatus' not in df.columns:
        print("  ⚠ TrackStatus column not found, skipping...\n")
        return df
    
    initial_count = len(df)
    
    print(f"Filtering for green flag only (TrackStatus = {TRACK_STATUS_GREEN})")
    
    # Keep only green flag laps
    df_clean = df[df['TrackStatus'] == TRACK_STATUS_GREEN].copy()
    
    removed = initial_count - len(df_clean)
    print(f"\nResults:")
    print(f"  Removed: {removed:,} laps ({removed/initial_count*100:.2f}%)")
    print(f"  Remaining: {len(df_clean):,} laps ({len(df_clean)/initial_count*100:.2f}%)\n")
    
    return df_clean

def filter_timing_accuracy(df):
    """Keep only accurate timing data"""
    print("=" * 80)
    print("STEP 3: FILTERING TIMING ACCURACY")
    print("=" * 80 + "\n")
    
    if 'IsAccurate' not in df.columns:
        print("  ⚠ IsAccurate column not found, skipping...\n")
        return df
    
    initial_count = len(df)
    
    print("Filtering for accurate timing only (IsAccurate = 1)")
    
    # Keep only accurate laps
    df_clean = df[df['IsAccurate'] == 1].copy()
    
    removed = initial_count - len(df_clean)
    print(f"\nResults:")
    print(f"  Removed: {removed:,} laps ({removed/initial_count*100:.2f}%)")
    print(f"  Remaining: {len(df_clean):,} laps ({len(df_clean)/initial_count*100:.2f}%)\n")
    
    return df_clean

def remove_first_last_laps(df):
    """Remove first and last laps of each race"""
    print("=" * 80)
    print("STEP 4: REMOVING FIRST/LAST RACE LAPS")
    print("=" * 80 + "\n")
    
    initial_count = len(df)
    
    print("Removing:")
    print("  - First lap (formation lap/start chaos)")
    print("  - Last lap (cool-down lap)\n")
    
    # Calculate min and max lap per race
    race_laps = df.groupby(['Year', 'Round'])['LapNumber'].agg(['min', 'max']).reset_index()
    race_laps.columns = ['Year', 'Round', 'FirstLap', 'LastLap']
    
    # Merge back
    df_merged = df.merge(race_laps, on=['Year', 'Round'], how='left')
    
    # Filter out first and last laps
    df_clean = df_merged[
        (df_merged['LapNumber'] != df_merged['FirstLap']) & 
        (df_merged['LapNumber'] != df_merged['LastLap'])
    ].copy()
    
    # Drop helper columns
    df_clean = df_clean.drop(['FirstLap', 'LastLap'], axis=1)
    
    removed = initial_count - len(df_clean)
    print(f"Results:")
    print(f"  Removed: {removed:,} laps ({removed/initial_count*100:.2f}%)")
    print(f"  Remaining: {len(df_clean):,} laps ({len(df_clean)/initial_count*100:.2f}%)\n")
    
    return df_clean

def remove_first_stint_laps(df):
    """Remove first lap of each tire stint (cold tires)"""
    print("=" * 80)
    print("STEP 5: REMOVING FIRST STINT LAPS")
    print("=" * 80 + "\n")
    
    if 'Stint' not in df.columns:
        print("  ⚠ Stint column not found, skipping...\n")
        return df
    
    initial_count = len(df)
    
    print(f"Removing first {MIN_STINT_LAP - 1} lap(s) of each stint (cold tires)")
    
    # Sort by driver, race, and lap
    df_sorted = df.sort_values(['Year', 'Round', 'Driver', 'Stint', 'LapNumber']).copy()
    
    # Calculate lap number within each stint
    df_sorted['StintLapNumber'] = df_sorted.groupby(['Year', 'Round', 'Driver', 'Stint']).cumcount() + 1
    
    # Keep only laps from MIN_STINT_LAP onwards
    df_clean = df_sorted[df_sorted['StintLapNumber'] >= MIN_STINT_LAP].copy()
    
    # Drop helper column
    df_clean = df_clean.drop('StintLapNumber', axis=1)
    
    removed = initial_count - len(df_clean)
    print(f"\nResults:")
    print(f"  Removed: {removed:,} laps ({removed/initial_count*100:.2f}%)")
    print(f"  Remaining: {len(df_clean):,} laps ({len(df_clean)/initial_count*100:.2f}%)\n")
    
    return df_clean

def remove_sector_outliers(df):
    """Remove laps with outlier sector times"""
    print("=" * 80)
    print("STEP 6: REMOVING SECTOR TIME OUTLIERS")
    print("=" * 80 + "\n")
    
    sector_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
    
    # Check if sector columns exist
    missing_sectors = [col for col in sector_cols if col not in df.columns]
    if missing_sectors:
        print(f"  ⚠ Missing sector columns: {missing_sectors}, skipping...\n")
        return df
    
    initial_count = len(df)
    
    print("Removing laps with outlier sector times (> 60s per sector)")
    
    # Filter: Each sector should be reasonable
    df_clean = df[
        (df['Sector1Time'] < 60) & 
        (df['Sector2Time'] < 60) & 
        (df['Sector3Time'] < 60) &
        (df['Sector1Time'] > 10) &
        (df['Sector2Time'] > 10) &
        (df['Sector3Time'] > 10)
    ].copy()
    
    removed = initial_count - len(df_clean)
    print(f"\nResults:")
    print(f"  Removed: {removed:,} laps ({removed/initial_count*100:.2f}%)")
    print(f"  Remaining: {len(df_clean):,} laps ({len(df_clean)/initial_count*100:.2f}%)\n")
    
    return df_clean

def validate_v3_quality(df, df_original):
    """Validate data quality after cleaning"""
    print("=" * 80)
    print("V3 DATA QUALITY VALIDATION")
    print("=" * 80 + "\n")
    
    print("Cleaning Summary:")
    print(f"  Original rows: {len(df_original):,}")
    print(f"  Cleaned rows: {len(df):,}")
    print(f"  Removed: {len(df_original) - len(df):,} ({(len(df_original) - len(df))/len(df_original)*100:.2f}%)")
    print(f"  Retention: {len(df)/len(df_original)*100:.2f}%\n")
    
    print("V3 Lap Time Distribution:")
    print(f"  Min: {df['LapTime'].min():.2f}s")
    print(f"  Max: {df['LapTime'].max():.2f}s")
    print(f"  Mean: {df['LapTime'].mean():.2f}s")
    print(f"  Median: {df['LapTime'].median():.2f}s")
    print(f"  Std: {df['LapTime'].std():.2f}s\n")
    
    # Check critical correlations
    print("Critical Correlation Check:")
    
    if 'TyreLife' in df.columns and 'LapTime' in df.columns:
        corr_v1 = df[['TyreLife', 'LapTime']].corr().iloc[0, 1]
        print(f"  TyreLife ↔ LapTime: {corr_v1:+.4f}", end="")
        if corr_v1 > 0:
            print(" ✅ POSITIVE! (Expected)")
        elif corr_v1 > -0.1:
            print(" ⚠️ Near zero (improvement from V2)")
        else:
            print(" ❌ Still negative (cleaning may need tuning)")
    
    if 'TyreLife' in df.columns and 'LapTime_FuelCorrected' in df.columns:
        corr_v2 = df[['TyreLife', 'LapTime_FuelCorrected']].corr().iloc[0, 1]
        print(f"  TyreLife ↔ LapTime_FuelCorrected: {corr_v2:+.4f}", end="")
        if corr_v2 > 0:
            print(" ✅ POSITIVE! (V3 SUCCESS!)")
        elif corr_v2 > -0.1:
            print(" ⚠️ Near zero (partial improvement)")
        else:
            print(" ❌ Still negative")
    
    if 'FuelRemaining_kg' in df.columns and 'LapTime' in df.columns:
        corr_fuel = df[['FuelRemaining_kg', 'LapTime']].corr().iloc[0, 1]
        print(f"  FuelRemaining ↔ LapTime: {corr_fuel:+.4f}", end="")
        if corr_fuel < -0.3:
            print(" ✅ Strong negative (heavy = slow)")
        elif corr_fuel < 0:
            print(" ⚠️ Weak negative")
        else:
            print(" ❌ Positive (wrong direction)")
    
    print()

def save_output(df):
    """Save V3 cleaned dataset"""
    print("=" * 80)
    print("SAVING V3 DATASET")
    print("=" * 80 + "\n")
    
    print(f"Output file: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    
    file_size = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}\n")
    
    print("V3 Features (same as V2):")
    print(f"  Core: Driver, Team, Circuit, LapNumber, Position, etc.")
    print(f"  Telemetry: LapTime, Sector times, Speed")
    print(f"  Tires: Compound, TyreLife, FreshTyre")
    print(f"  Weather: Temperature, Wind, Precipitation")
    print(f"  V2 Physics: NormalizedLap, FuelRemaining, LapTime_FuelCorrected")
    print(f"  V2 Traffic: InTraffic, GapToCarAhead")
    print(f"\n  ✅ All V2 features preserved")
    print(f"  ✅ Data quality significantly improved")
    
    print("\n" + "=" * 80)
    print("V3 CLEANING COMPLETE")
    print("=" * 80)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution"""
    try:
        # Load data
        df_original = load_data()
        
        # Analyze initial quality
        analyze_initial_quality(df_original)
        
        # Apply cleaning steps
        df = df_original.copy()
        df = remove_outlier_lap_times(df)
        df = filter_track_status(df)
        df = filter_timing_accuracy(df)
        df = remove_first_last_laps(df)
        df = remove_first_stint_laps(df)
        df = remove_sector_outliers(df)
        
        # Validate
        validate_v3_quality(df, df_original)
        
        # Save
        save_output(df)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR")
        print("=" * 80)
        print(f"{type(e).__name__}: {e}")
        print("\nProcess failed.")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
