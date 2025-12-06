"""
F1 Track Enrichment V3.1 - Real Circuit Characteristics

V3.1 Critical Goal: Add REAL track properties to fix track-normalized analysis

Why V3 Failed (Partially):
- V3 cleaned data but kept placeholder TrackLength_km = 4.5
- Monaco 20 laps ≠ Spa 20 laps (3.3km vs 7.0km)
- TyreLife not normalized by track characteristics
- This confounds degradation analysis!

V3.1 Solution:
1. Load LabelEncoder to decode Circuit IDs
2. Map Circuit IDs to F1DB circuit names
3. Extract real track properties:
   - Actual circuit length (km)
   - Number of turns/corners
   - Circuit type (RACE/STREET/ROAD)
   - Track direction (CLOCKWISE/ANTI_CLOCKWISE)
4. Create track-normalized features:
   - TyreLife_PerKm = TyreLife × TrackLength
   - CornerDensity = Turns / TrackLength
   - LapTime_PerKm = LapTime / TrackLength

Expected Outcome:
- TyreLife correlation should improve when normalized by track length
- 20 laps at Monaco (3.3km) = 66km vs 20 laps at Spa (7.0km) = 140km
- Corner-heavy tracks (Monaco) vs high-speed tracks (Monza)
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'f1_training_data_v3.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'f1_training_data_v3_1.csv')
ENCODERS_FILE = os.path.join(BASE_DIR, 'data', 'f1_encoders.pkl')
F1DB_CIRCUITS = os.path.join(BASE_DIR, 'data', 'f1db-csv', 'f1db-circuits.csv')

def load_encoders():
    """Load LabelEncoders from V1 preprocessing"""
    print("=" * 80)
    print("F1 TRACK ENRICHMENT V3.1")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not os.path.exists(ENCODERS_FILE):
        print("⚠️  ERROR: f1_encoders.pkl not found!")
        print(f"   Expected: {ENCODERS_FILE}")
        print(f"   This file should have been created during V1 preprocessing")
        print(f"   Need to run: src/f1_preprocessing.py first\n")
        return None
    
    print(f"Loading encoders: {ENCODERS_FILE}")
    encoders = joblib.load(ENCODERS_FILE)
    print(f"  ✓ Loaded {len(encoders)} encoders")
    print(f"  ✓ Available: {list(encoders.keys())}\n")
    
    return encoders

def load_f1db_circuits():
    """Load F1DB circuit characteristics"""
    print(f"Loading F1DB circuits: {F1DB_CIRCUITS}")
    circuits_df = pd.read_csv(F1DB_CIRCUITS)
    print(f"  ✓ Loaded {len(circuits_df)} circuits from F1DB")
    print(f"  ✓ Columns: {list(circuits_df.columns)}\n")
    
    return circuits_df

def decode_circuits(df, encoders):
    """Decode Circuit column using LabelEncoder"""
    print("STEP 1: Decode Circuit IDs")
    print("-" * 80)
    
    if 'Circuit' not in encoders:
        print("⚠️  ERROR: 'Circuit' encoder not found!")
        print(f"   Available encoders: {list(encoders.keys())}")
        return df
    
    circuit_encoder = encoders['Circuit']
    
    # Create decoded circuit names
    print(f"Decoding {len(df)} laps...")
    df['CircuitName'] = circuit_encoder.inverse_transform(df['Circuit'])
    
    # Show unique circuits
    unique_circuits = df['CircuitName'].value_counts()
    print(f"  ✓ Found {len(unique_circuits)} unique circuits")
    print(f"  ✓ Top 5 circuits:")
    for circuit, count in unique_circuits.head(5).items():
        print(f"     - {circuit}: {count} laps")
    
    print(f"  ✓ Decoded column created: CircuitName\n")
    return df

def map_circuit_characteristics(df, circuits_df):
    """Map F1DB circuit characteristics to dataset"""
    print("STEP 2: Map Circuit Characteristics from F1DB")
    print("-" * 80)
    
    # F1DB uses circuit 'id' (lowercase, hyphenated)
    # FastF1 uses circuit 'name' (mixed case, spaces)
    # Need to create mapping
    
    # Create circuit ID mapping (manual for known circuits)
    circuit_mapping = {
        'Albert Park Grand Prix Circuit': 'melbourne',
        'Bahrain International Circuit': 'bahrain',
        'Shanghai International Circuit': 'shanghai',
        'Baku City Circuit': 'baku',
        'Circuit de Barcelona-Catalunya': 'catalunya',
        'Circuit de Monaco': 'monaco',
        'Circuit Gilles Villeneuve': 'montreal',
        'Circuit Paul Ricard': 'paul-ricard',
        'Red Bull Ring': 'a1-ring',
        'Silverstone Circuit': 'silverstone',
        'Hockenheimring': 'hockenheimring',
        'Hungaroring': 'hungaroring',
        'Circuit de Spa-Francorchamps': 'spa',
        'Autodromo Nazionale di Monza': 'monza',
        'Marina Bay Street Circuit': 'marina-bay',
        'Sochi Autodrom': 'sochi',
        'Suzuka Circuit': 'suzuka',
        'Autódromo Hermanos Rodríguez': 'rodriguez',
        'Circuit of the Americas': 'austin',
        'Autódromo José Carlos Pace': 'interlagos',
        'Yas Marina Circuit': 'yas-marina',
        'Autodromo Internazionale Enzo e Dino Ferrari': 'imola',
        'Portimão Circuit': 'portimao',
        'Istanbul Park': 'istanbul',
        'Jeddah Corniche Circuit': 'jeddah',
        'Miami International Autodrome': 'miami',
        'Losail International Circuit': 'losail',
        'Las Vegas Street Circuit': 'las-vegas',
        'Autódromo Internacional do Algarve': 'portimao',
        'Nürburgring': 'nurburgring',
        'Circuit Zandvoort': 'zandvoort',
        'Korea International Circuit': 'korea',
        'Buddh International Circuit': 'buddh',
        'Sepang International Circuit': 'sepang'
    }
    
    print(f"Mapping {len(df['CircuitName'].unique())} unique circuits...")
    
    # Map circuit IDs
    df['CircuitID'] = df['CircuitName'].map(circuit_mapping)
    
    # Check unmapped circuits
    unmapped = df[df['CircuitID'].isna()]['CircuitName'].unique()
    if len(unmapped) > 0:
        print(f"  ⚠️  {len(unmapped)} circuits not mapped:")
        for circuit in unmapped[:5]:
            print(f"     - {circuit}")
    
    # Merge with F1DB data
    circuits_df = circuits_df.rename(columns={'id': 'CircuitID'})
    
    # Select relevant columns
    circuit_cols = ['CircuitID', 'length', 'turns', 'type', 'direction']
    circuits_subset = circuits_df[circuit_cols].copy()
    
    # Merge
    print(f"Merging circuit characteristics...")
    df = df.merge(circuits_subset, on='CircuitID', how='left')
    
    # Rename columns
    df = df.rename(columns={
        'length': 'TrackLength_Real_km',
        'turns': 'TrackTurns',
        'type': 'TrackType_Real',
        'direction': 'TrackDirection'
    })
    
    # Show merge results
    matched = df['TrackLength_Real_km'].notna().sum()
    total = len(df)
    match_rate = (matched / total) * 100
    
    print(f"  ✓ Matched {matched:,} / {total:,} laps ({match_rate:.1f}%)")
    print(f"  ✓ New columns: TrackLength_Real_km, TrackTurns, TrackType_Real, TrackDirection\n")
    
    return df

def create_track_normalized_features(df):
    """Create track-normalized features"""
    print("STEP 3: Create Track-Normalized Features")
    print("-" * 80)
    
    # Only process laps with real track data
    valid_mask = df['TrackLength_Real_km'].notna()
    valid_count = valid_mask.sum()
    
    print(f"Processing {valid_count:,} laps with valid track data...")
    
    # Feature 1: Total Distance on Current Tires (km)
    df.loc[valid_mask, 'TyreDistance_km'] = (
        df.loc[valid_mask, 'TyreLife'] * df.loc[valid_mask, 'TrackLength_Real_km']
    )
    
    # Feature 2: Lap Time per Kilometer (sec/km)
    df.loc[valid_mask, 'LapTime_PerKm'] = (
        df.loc[valid_mask, 'LapTime'] / df.loc[valid_mask, 'TrackLength_Real_km']
    )
    
    # Feature 3: Corner Density (corners/km)
    df.loc[valid_mask, 'CornerDensity'] = (
        df.loc[valid_mask, 'TrackTurns'] / df.loc[valid_mask, 'TrackLength_Real_km']
    )
    
    # Feature 4: Fuel-Corrected Lap Time per Kilometer
    if 'LapTime_FuelCorrected' in df.columns:
        df.loc[valid_mask, 'LapTime_FuelCorrected_PerKm'] = (
            df.loc[valid_mask, 'LapTime_FuelCorrected'] / df.loc[valid_mask, 'TrackLength_Real_km']
        )
    
    print(f"  ✓ TyreDistance_km: Distance traveled on current tires")
    print(f"     Range: {df['TyreDistance_km'].min():.1f} - {df['TyreDistance_km'].max():.1f} km")
    
    print(f"  ✓ LapTime_PerKm: Lap time normalized by track length")
    print(f"     Range: {df['LapTime_PerKm'].min():.1f} - {df['LapTime_PerKm'].max():.1f} sec/km")
    
    print(f"  ✓ CornerDensity: Corners per kilometer")
    print(f"     Range: {df['CornerDensity'].min():.2f} - {df['CornerDensity'].max():.2f} corners/km")
    
    if 'LapTime_FuelCorrected_PerKm' in df.columns:
        print(f"  ✓ LapTime_FuelCorrected_PerKm: Fuel-corrected, track-normalized")
        print(f"     Range: {df['LapTime_FuelCorrected_PerKm'].min():.1f} - {df['LapTime_FuelCorrected_PerKm'].max():.1f} sec/km\n")
    
    return df

def validate_track_features(df):
    """Validate track-normalized features"""
    print("STEP 4: Validate Track-Normalized Features")
    print("-" * 80)
    
    # Check for real track data
    has_real_data = df['TrackLength_Real_km'].notna()
    real_count = has_real_data.sum()
    total_count = len(df)
    
    print(f"Real Track Data Coverage:")
    print(f"  - Laps with real data: {real_count:,} ({(real_count/total_count)*100:.1f}%)")
    print(f"  - Laps missing data: {total_count - real_count:,} ({((total_count-real_count)/total_count)*100:.1f}%)\n")
    
    # Show track length comparison
    print(f"Track Length Comparison (Old vs Real):")
    comparison_df = df[has_real_data].groupby('CircuitName').agg({
        'TrackLength_km': 'first',
        'TrackLength_Real_km': 'first',
        'TrackTurns': 'first',
        'CornerDensity': 'first'
    }).round(3)
    
    comparison_df['Length_Diff'] = (
        comparison_df['TrackLength_Real_km'] - comparison_df['TrackLength_km']
    ).round(3)
    
    print(comparison_df.head(10).to_string())
    print()
    
    # Correlation check: TyreDistance vs LapTime
    if 'TyreDistance_km' in df.columns:
        corr_dist = df['TyreDistance_km'].corr(df['LapTime'])
        corr_dist_fuel = df['TyreDistance_km'].corr(df['LapTime_FuelCorrected'])
        
        print(f"NEW Correlation Check (Track-Normalized):")
        print(f"  - TyreDistance_km ↔ LapTime: {corr_dist:+.4f}")
        print(f"  - TyreDistance_km ↔ LapTime_FuelCorrected: {corr_dist_fuel:+.4f}")
        
        # Compare to old correlation
        corr_life = df['TyreLife'].corr(df['LapTime'])
        corr_life_fuel = df['TyreLife'].corr(df['LapTime_FuelCorrected'])
        
        print(f"\nOLD Correlation (Not Track-Normalized):")
        print(f"  - TyreLife ↔ LapTime: {corr_life:+.4f}")
        print(f"  - TyreLife ↔ LapTime_FuelCorrected: {corr_life_fuel:+.4f}")
        
        print(f"\nImprovement:")
        print(f"  - Raw lap time: {corr_dist - corr_life:+.4f}")
        print(f"  - Fuel-corrected: {corr_dist_fuel - corr_life_fuel:+.4f}\n")
        
        if corr_dist_fuel > 0:
            print("  🎉 SUCCESS! Track-normalized correlation is POSITIVE!")
        else:
            print("  ⚠️  Still negative, but track normalization helps")
    
    return df

def save_v3_1_data(df):
    """Save V3.1 dataset"""
    print("STEP 5: Save V3.1 Dataset")
    print("-" * 80)
    
    print(f"Saving to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"  ✓ File saved: {file_size:.2f} MB")
    print(f"  ✓ Rows: {len(df):,}")
    print(f"  ✓ Columns: {len(df.columns)} (V3: 48 → V3.1: {len(df.columns)})")
    
    # Show new columns
    new_cols = [col for col in df.columns if col not in [
        'Year', 'Round', 'Driver', 'LapNumber', 'LapTime', 'Sector1Time',
        'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife', 'Stint',
        'Circuit', 'Weather', 'TrackStatus', 'TrackLength_km', 'TrackType'
    ]]
    
    print(f"\n  New V3.1 Columns:")
    for col in new_cols[-10:]:  # Show last 10 new columns
        non_null = df[col].notna().sum()
        print(f"     - {col}: {non_null:,} non-null values")
    
    print(f"\n{'=' * 80}")
    print(f"V3.1 TRACK ENRICHMENT COMPLETE")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")
    
    return df

def main():
    """Main execution"""
    try:
        # Load encoders
        encoders = load_encoders()
        if encoders is None:
            return
        
        # Load F1DB circuits
        circuits_df = load_f1db_circuits()
        
        # Load V3 data
        print(f"Loading V3 data: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        print(f"  ✓ Loaded {len(df):,} laps, {len(df.columns)} columns\n")
        
        # Decode circuits
        df = decode_circuits(df, encoders)
        
        # Map circuit characteristics
        df = map_circuit_characteristics(df, circuits_df)
        
        # Create track-normalized features
        df = create_track_normalized_features(df)
        
        # Validate
        df = validate_track_features(df)
        
        # Save
        df = save_v3_1_data(df)
        
        print("✅ V3.1 Track enrichment successful!")
        print(f"   Next: Run analysis to see if track normalization fixes correlations")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
