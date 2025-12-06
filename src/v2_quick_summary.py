"""Quick V2 summary report"""
import pandas as pd

df = pd.read_csv(r'c:\Users\isml_\Desktop\Güz dönemi 2025-2026\Yapay sinir ağları\Yapay sinir ağları projesi\data\f1_training_data_v2.csv')

print('='*80)
print('F1 V2 DATASET SUMMARY')
print('='*80)
print(f'\nTotal rows: {len(df):,}')
print(f'Total columns: {len(df.columns)}')
print(f'Memory: {df.memory_usage(deep=True).sum()/1024**2:.2f} MB')

print('\n' + '='*80)
print('V2 NEW FEATURES')
print('='*80)

v2_cols = ['NormalizedLap', 'RaceProgress_pct', 'TotalRaceLaps', 
           'FuelRemaining_kg', 'FuelPenalty_sec', 'LapTime_FuelCorrected', 
           'GapToCarAhead_sec', 'InTraffic', 'TrackLength_km', 'TrackType']

for col in v2_cols:
    if col in df.columns:
        if df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
            print(f'{col:25s}: {str(df[col].dtype):10s} | Range: {df[col].min():8.2f} - {df[col].max():8.2f}')
        else:
            unique = df[col].nunique()
            print(f'{col:25s}: {str(df[col].dtype):10s} | {unique} unique values')

print('\n' + '='*80)
print('KEY CORRELATIONS - V1 vs V2 COMPARISON')
print('='*80)

corr_cols = ['TyreLife', 'LapTime', 'LapTime_FuelCorrected', 'FuelRemaining_kg', 
             'NormalizedLap', 'InTraffic', 'Position']
corr = df[corr_cols].corr()

print('\n🔴 CRITICAL: TyreLife Correlation (Expected: POSITIVE after fuel correction)')
print(f'  V1 (Raw):           TyreLife ↔ LapTime               = {corr.loc["TyreLife", "LapTime"]:+.4f} ❌')
print(f'  V2 (Fuel-corrected): TyreLife ↔ LapTime_FuelCorrected = {corr.loc["TyreLife", "LapTime_FuelCorrected"]:+.4f}', end='')
if corr.loc["TyreLife", "LapTime_FuelCorrected"] > 0:
    print(' ✅ FIXED!')
else:
    print(' ⚠️ Still negative')

print('\n🔥 Fuel Effect (Expected: NEGATIVE - heavier = slower)')
print(f'  FuelRemaining ↔ LapTime = {corr.loc["FuelRemaining_kg", "LapTime"]:+.4f}', end='')
if corr.loc["FuelRemaining_kg", "LapTime"] < -0.3:
    print(' ✅')
elif corr.loc["FuelRemaining_kg", "LapTime"] < 0:
    print(' ⚠️ Weak')
else:
    print(' ❌ WRONG!')

print('\n📊 Normalization (Expected: -1.0)')
print(f'  NormalizedLap ↔ FuelRemaining = {corr.loc["NormalizedLap", "FuelRemaining_kg"]:+.4f}', end='')
if abs(corr.loc["NormalizedLap", "FuelRemaining_kg"] + 1.0) < 0.01:
    print(' ✅ Perfect!')
else:
    print(' ⚠️')

print('\n🚗 Traffic Effect (Expected: POSITIVE - traffic slows laps)')
print(f'  InTraffic ↔ LapTime = {corr.loc["InTraffic", "LapTime"]:+.4f}', end='')
if corr.loc["InTraffic", "LapTime"] > 0:
    print(' ✅')
else:
    print(' ❌')

print('\n' + '='*80)
print('DEGRADATION ANALYSIS')
print('='*80)

# Tire degradation by compound
print('\nAverage degradation by tire compound:')
tire_deg = df.groupby('Compound').agg({
    'TyreLife': 'mean',
    'LapTime_FuelCorrected': 'mean'
}).round(2)
print(tire_deg)

# Traffic impact
print('\nTraffic impact on lap times:')
traffic_impact = df.groupby('InTraffic')[['LapTime', 'LapTime_FuelCorrected']].mean()
print(f'  Free air (InTraffic=0): {traffic_impact.loc[0, "LapTime_FuelCorrected"]:.3f}s')
print(f'  In traffic (InTraffic=1): {traffic_impact.loc[1, "LapTime_FuelCorrected"]:.3f}s')
penalty = traffic_impact.loc[1, "LapTime_FuelCorrected"] - traffic_impact.loc[0, "LapTime_FuelCorrected"]
print(f'  Penalty: {penalty:+.3f}s')

print('\n' + '='*80)
print('VISUALIZATION FILES CREATED')
print('='*80)
print('  ✅ f1_correlation_matrix_v2.png')
print('  ✅ f1_fuel_analysis_v2.png')
print('  ✅ f1_tire_degradation_v2.png')
print('  ✅ f1_traffic_analysis_v2.png')
print('  ✅ f1_track_comparison_v2.png')
print('='*80)
