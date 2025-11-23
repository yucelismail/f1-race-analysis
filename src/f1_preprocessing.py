# -*- coding: utf-8 -*-
"""
F1 Machine Learning Preprocessing
Prepares enriched F1 dataset for ML models.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Configuration
INPUT_FILE = '../data/f1_complete_dataset.csv'
OUTPUT_FILE = '../data/f1_training_data.csv'
ENCODERS_FILE = '../data/f1_encoders.pkl'

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("="*80)
print("F1 Machine Learning Preprocessing")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# Step 1: Load Data
print("Step 1: Loading data...")

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found!")
    exit(1)

df = pd.read_csv(INPUT_FILE)
initial_rows = len(df)
initial_cols = len(df.columns)

print(f"  * Loaded: {len(df):,} rows x {len(df.columns)} columns")
print(f"  * File size: {os.path.getsize(INPUT_FILE) / (1024*1024):.2f} MB\n")


# Step 2: Remove Unnecessary Columns
print("Step 2: Removing unnecessary columns...")

columns_to_drop = [
    'DriverName', 'DriverAbbr', 'DriverFullName',
    'CircuitName', 'CircuitLocation',
    'Country', 'Location',
    'PitOutTime', 'PitInTime',
    'DriverNumber',
]

existing_drops = [col for col in columns_to_drop if col in df.columns]

if existing_drops:
    df = df.drop(columns=existing_drops)
    print(f"  * {len(existing_drops)} columns removed:")
    for col in existing_drops[:5]:
        print(f"     - {col}")
    if len(existing_drops) > 5:
        print(f"     ... and {len(existing_drops) - 5} more")
    print(f"  * Remaining columns: {len(df.columns)}\n")
else:
    print("  * No columns to remove\n")


# Step 3: Missing Data Imputation
print("Step 3: Filling missing data...")

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


# Step 4: Categorical Encoding (Label Encoding)
print("Step 4: Encoding categorical data...")

categorical_columns = ['Driver', 'Team', 'Circuit', 'Compound', 'Status']
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

print(f"  * Total {len(encoders)} encoders saved\n")


# Step 5: Boolean Conversion
print("Step 5: Converting boolean columns...")

boolean_columns = ['FreshTyre', 'IsAccurate', 'Rainfall']

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

print()


# Step 6: Data Type Optimization
print("Step 6: Optimizing data types...")

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


# Step 7: Final Checks
print("Step 7: Final checks...")

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

# Data type summary
print(f"\n  Data type distribution:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"     - {str(dtype):<10} : {count:>3} columns")

print()


# Step 8: Save
print("Step 8: Saving files...")

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


# Step 9: Summary Statistics
print("="*80)
print("Preprocessing Summary")
print("="*80)
print(f"  - Initial rows       : {initial_rows:,}")
print(f"  - Final rows         : {len(df):,}")
print(f"  - Removed rows       : {initial_rows - len(df):,} ({((initial_rows - len(df))/initial_rows)*100:.2f}%)")
print(f"  - Initial columns    : {initial_cols}")
print(f"  - Final columns      : {len(df.columns)}")
print(f"  - Removed columns    : {initial_cols - len(df.columns)}")
print(f"  - Total encoders     : {len(encoders)}")
print(f"  - Memory usage       : {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
print()


# Step 10: Data Preview
print("="*80)
print("Processed Data Preview")
print("="*80)
print("\nDataFrame Info:")
print("-"*80)
df.info()

print("\nFirst 5 Rows:")
print("-"*80)
preview_cols = ['Year', 'Round', 'Driver', 'Team', 'LapNumber', 'Position', 
                'LapTime', 'GridPosition', 'Status', 'Compound', 'TyreLife',
                'WeatherTemp', 'WeatherPrecipitation', 'Rainfall']
available_preview = [col for col in preview_cols if col in df.columns]

if available_preview:
    print(df[available_preview].head(5).to_string(index=False))
else:
    print(df.head(5))

print("\nBasic Statistics:")
print("-"*80)
stat_cols = ['LapTime', 'LapNumber', 'Position', 'SpeedST', 'TyreLife', 
             'WeatherTemp', 'WeatherPrecipitation', 'DriverAge']
available_stats = [col for col in stat_cols if col in df.columns]

if available_stats:
    print(df[available_stats].describe().to_string())

print("\n" + "="*80)
print("Preprocessing Complete")
print(f"Training data: {os.path.abspath(OUTPUT_FILE)}")
print(f"Encoder file: {os.path.abspath(ENCODERS_FILE)}")
print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")


# Encoder Usage Example
print("Encoder Usage Example")
print("="*80)

if encoders:
    print("To reload encoders:")
    print("   encoders = joblib.load('f1_encoders.pkl')")
    print()
    
    if 'Driver' in encoders:
        driver_encoder = encoders['Driver']
        print(f"Example: Which driver is ID=0?")
        print(f"   Answer: {driver_encoder.inverse_transform([0])[0]}")
        print()
        
        print(f"All Drivers ({len(driver_encoder.classes_)} drivers):")
        for idx, driver in enumerate(driver_encoder.classes_[:10]):
            print(f"   {idx:>2} -> {driver}")
        if len(driver_encoder.classes_) > 10:
            print(f"   ... and {len(driver_encoder.classes_) - 10} more")
        print()
    
    if 'Compound' in encoders:
        compound_encoder = encoders['Compound']
        print(f"Tire Compounds:")
        for idx, compound in enumerate(compound_encoder.classes_):
            print(f"   {idx} -> {compound}")

print("\n" + "="*80)
print("Next Steps:")
print("="*80)
print("  1. Data ready for ML - f1_training_data.csv")
print("  2. Feature Engineering (optional)")
print("  3. Train/Test split (80/20 or 70/15/15)")
print("  4. Feature Scaling (StandardScaler or MinMaxScaler)")
print("  5. Model Selection:")
print("     - XGBoost - Gradient Boosting (for tabular data)")
print("     - LSTM - Recurrent Neural Network (for sequence prediction)")
print("     - LightGBM - Fast gradient boosting")
print("  6. Hyperparameter Tuning (GridSearch, RandomSearch)")
print("  7. Model Evaluation (Accuracy, F1-Score, Confusion Matrix)")
print("="*80 + "\n")

print(f"Process completed successfully!")
print(f"{len(df):,} rows x {len(df.columns)} columns ready for training\n")
