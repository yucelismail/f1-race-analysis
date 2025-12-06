"""
F1 Data Analysis V2 - Enhanced with Fuel-Corrected Features

Key V2 additions:
- Fuel load modeling correlation analysis
- Normalized lap distance validation
- Traffic effect visualization
- Comparison of LapTime vs LapTime_FuelCorrected

Expected fixes:
- TyreLife correlation should now be POSITIVE (not negative)
- FuelRemaining should show strong negative correlation with LapTime
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# File paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'f1_training_data_v2.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

def load_data():
    """Load preprocessed V2 dataset"""
    print("=" * 80)
    print("F1 DATA ANALYSIS V2 - FUEL-CORRECTED FEATURES")
    print("=" * 80)
    
    df = pd.read_csv(INPUT_FILE)
    print(f"\nDataset loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def basic_statistics(df):
    """Display basic dataset statistics"""
    print("\n" + "=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)
    
    print(f"\nDate range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Total races: {df['Round'].nunique()}")
    print(f"Unique drivers: {df['Driver'].nunique()}")
    print(f"Unique teams: {df['Team'].nunique()}")
    print(f"Unique circuits: {df['Circuit'].nunique()}")
    
    print("\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values (100% complete)")
    else:
        print(missing[missing > 0])
    
    print("\nV2 New Features Coverage:")
    v2_features = ['NormalizedLap', 'RaceProgress_pct', 'FuelRemaining_kg', 
                   'FuelPenalty_sec', 'LapTime_FuelCorrected', 'GapToCarAhead_sec', 
                   'InTraffic', 'TrackLength_km', 'TrackType']
    for feat in v2_features:
        if feat in df.columns:
            coverage = (1 - df[feat].isnull().sum() / len(df)) * 100
            print(f"  {feat}: {coverage:.1f}% coverage")

def fuel_model_validation(df):
    """Validate fuel load modeling physics"""
    print("\n" + "=" * 80)
    print("FUEL MODEL VALIDATION")
    print("=" * 80)
    
    print("\nFuel Remaining Statistics:")
    print(f"  Min: {df['FuelRemaining_kg'].min():.1f} kg (end of race)")
    print(f"  Max: {df['FuelRemaining_kg'].max():.1f} kg (start of race)")
    print(f"  Mean: {df['FuelRemaining_kg'].mean():.1f} kg (mid-race)")
    
    print("\nFuel Penalty Statistics:")
    print(f"  Min: {df['FuelPenalty_sec'].min():.2f} seconds (light car)")
    print(f"  Max: {df['FuelPenalty_sec'].max():.2f} seconds (full tank)")
    print(f"  Mean: {df['FuelPenalty_sec'].mean():.2f} seconds")
    
    # Check linearity
    fuel_lap_corr = df[['NormalizedLap', 'FuelRemaining_kg']].corr().iloc[0, 1]
    print(f"\nNormalizedLap ↔ FuelRemaining correlation: {fuel_lap_corr:.3f}")
    if abs(fuel_lap_corr + 1.0) < 0.01:
        print("✓ Perfect linear burn model (correlation ≈ -1.0)")
    else:
        print("⚠ Fuel model deviation detected")

def traffic_analysis(df):
    """Analyze traffic detection statistics"""
    print("\n" + "=" * 80)
    print("TRAFFIC DETECTION ANALYSIS")
    print("=" * 80)
    
    traffic_pct = (df['InTraffic'].sum() / len(df)) * 100
    print(f"\nLaps in traffic (<1.5s gap): {traffic_pct:.1f}%")
    
    if 'GapToCarAhead_sec' in df.columns:
        clean_gaps = df[df['GapToCarAhead_sec'] > 0]['GapToCarAhead_sec']
        print(f"\nGap to car ahead statistics (excluding leaders):")
        print(f"  Mean: {clean_gaps.mean():.2f} seconds")
        print(f"  Median: {clean_gaps.median():.2f} seconds")
        print(f"  95th percentile: {clean_gaps.quantile(0.95):.2f} seconds")

def track_type_distribution(df):
    """Analyze track type classification"""
    print("\n" + "=" * 80)
    print("TRACK TYPE DISTRIBUTION")
    print("=" * 80)
    
    if 'TrackType' in df.columns:
        track_counts = df.groupby('TrackType')['Circuit'].nunique()
        print("\nCircuits by type:")
        for track_type, count in track_counts.items():
            print(f"  {track_type}: {count} circuits")
        
        if 'TrackLength_km' in df.columns:
            print("\nTrack length ranges:")
            for track_type in track_counts.index:
                lengths = df[df['TrackType'] == track_type]['TrackLength_km'].unique()
                print(f"  {track_type}: {lengths.min():.2f} - {lengths.max():.2f} km")

def correlation_analysis(df):
    """Generate comprehensive correlation matrix"""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS - V2 ENHANCED")
    print("=" * 80)
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude IDs and year
    exclude_cols = ['Year', 'Round', 'Driver', 'Team', 'Circuit', 'Compound']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Print key correlations
    print("\nCRITICAL V2 CORRELATIONS:")
    print("-" * 80)
    
    # Check TyreLife correlation fix
    if 'TyreLife' in numeric_cols:
        tyre_laptime = corr_matrix.loc['TyreLife', 'LapTime']
        print(f"\n1. TyreLife ↔ LapTime (V1): {tyre_laptime:+.3f}")
        
        if 'LapTime_FuelCorrected' in numeric_cols:
            tyre_corrected = corr_matrix.loc['TyreLife', 'LapTime_FuelCorrected']
            print(f"   TyreLife ↔ LapTime_FuelCorrected (V2): {tyre_corrected:+.3f}")
            
            if tyre_laptime < 0 and tyre_corrected > 0:
                print("   ✓ CORRELATION FIX SUCCESSFUL!")
                print("   ✓ Fuel masking removed, tire degradation signal recovered")
            elif tyre_corrected > 0:
                print("   ✓ Positive correlation confirmed (physically correct)")
            else:
                print("   ⚠ Still negative - fuel model may need tuning")
    
    # Fuel effect validation
    if 'FuelRemaining_kg' in numeric_cols:
        fuel_laptime = corr_matrix.loc['FuelRemaining_kg', 'LapTime']
        print(f"\n2. FuelRemaining_kg ↔ LapTime: {fuel_laptime:+.3f}")
        if fuel_laptime < -0.4:
            print("   ✓ Strong negative correlation (heavier = slower)")
        else:
            print("   ⚠ Weak fuel effect detected")
    
    # Normalization validation
    if 'NormalizedLap' in numeric_cols and 'FuelRemaining_kg' in numeric_cols:
        norm_fuel = corr_matrix.loc['NormalizedLap', 'FuelRemaining_kg']
        print(f"\n3. NormalizedLap ↔ FuelRemaining_kg: {norm_fuel:+.3f}")
        if abs(norm_fuel + 1.0) < 0.05:
            print("   ✓ Perfect linear model (≈ -1.0)")
    
    # Traffic effect
    if 'InTraffic' in numeric_cols:
        traffic_laptime = corr_matrix.loc['InTraffic', 'LapTime']
        print(f"\n4. InTraffic ↔ LapTime: {traffic_laptime:+.3f}")
        if traffic_laptime > 0:
            print("   ✓ Positive correlation (traffic slows laps)")
    
    # Track length effect
    if 'TrackLength_km' in numeric_cols:
        length_laptime = corr_matrix.loc['TrackLength_km', 'LapTime']
        print(f"\n5. TrackLength_km ↔ LapTime: {length_laptime:+.3f}")
        if length_laptime > 0.5:
            print("   ✓ Strong positive correlation (longer tracks = slower laps)")
    
    # Plot correlation matrix
    print("\nGenerating correlation heatmap...")
    plt.figure(figsize=(20, 16))
    
    # Select top correlations with LapTime
    if 'LapTime' in corr_matrix.columns:
        laptime_corrs = corr_matrix['LapTime'].abs().sort_values(ascending=False).head(20).index
        subset_corr = corr_matrix.loc[laptime_corrs, laptime_corrs]
        
        sns.heatmap(subset_corr, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True, linewidths=0.5,
                    cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('F1 V2 Correlation Matrix - Top 20 LapTime Correlations', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/f1_correlation_matrix_v2.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR}/f1_correlation_matrix_v2.png")
        plt.close()

def fuel_effect_visualization(df):
    """Visualize fuel load effect on lap times"""
    print("\n" + "=" * 80)
    print("FUEL EFFECT VISUALIZATION")
    print("=" * 80)
    
    if 'FuelRemaining_kg' not in df.columns or 'LapTime' not in df.columns:
        print("⚠ Required columns missing")
        return
    
    # Sample data to avoid overplotting
    sample_df = df.sample(min(50000, len(df)), random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Fuel remaining vs lap time
    axes[0, 0].scatter(sample_df['FuelRemaining_kg'], sample_df['LapTime'], 
                       alpha=0.3, s=1, c='steelblue')
    axes[0, 0].set_xlabel('Fuel Remaining (kg)', fontsize=12)
    axes[0, 0].set_ylabel('Lap Time (seconds)', fontsize=12)
    axes[0, 0].set_title('Fuel Load Effect on Lap Time', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Normalized lap vs fuel remaining
    axes[0, 1].scatter(sample_df['NormalizedLap'], sample_df['FuelRemaining_kg'], 
                       alpha=0.3, s=1, c='darkorange')
    axes[0, 1].set_xlabel('Normalized Lap (0-1)', fontsize=12)
    axes[0, 1].set_ylabel('Fuel Remaining (kg)', fontsize=12)
    axes[0, 1].set_title('Fuel Burn Linearity', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Fuel penalty distribution
    axes[1, 0].hist(sample_df['FuelPenalty_sec'], bins=50, color='crimson', 
                    alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Fuel Penalty (seconds)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Fuel Penalty Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: LapTime vs LapTime_FuelCorrected comparison
    if 'LapTime_FuelCorrected' in sample_df.columns:
        axes[1, 1].scatter(sample_df['LapTime'], sample_df['LapTime_FuelCorrected'], 
                           alpha=0.3, s=1, c='forestgreen')
        axes[1, 1].plot([sample_df['LapTime'].min(), sample_df['LapTime'].max()],
                        [sample_df['LapTime'].min(), sample_df['LapTime'].max()],
                        'r--', linewidth=2, label='y=x')
        axes[1, 1].set_xlabel('Raw Lap Time (seconds)', fontsize=12)
        axes[1, 1].set_ylabel('Fuel-Corrected Lap Time (seconds)', fontsize=12)
        axes[1, 1].set_title('Lap Time Correction Effect', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/f1_fuel_analysis_v2.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/f1_fuel_analysis_v2.png")
    plt.close()

def tire_degradation_analysis(df):
    """Analyze tire degradation with fuel correction"""
    print("\n" + "=" * 80)
    print("TIRE DEGRADATION ANALYSIS (FUEL-CORRECTED)")
    print("=" * 80)
    
    if 'TyreLife' not in df.columns:
        print("⚠ TyreLife column missing")
        return
    
    # Group by tire life
    tire_groups = df.groupby('TyreLife').agg({
        'LapTime': 'mean',
        'LapTime_FuelCorrected': 'mean' if 'LapTime_FuelCorrected' in df.columns else 'mean'
    }).reset_index()
    
    # Filter outliers (tire life 1-50 laps)
    tire_groups = tire_groups[(tire_groups['TyreLife'] >= 1) & (tire_groups['TyreLife'] <= 50)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot raw lap time
    ax.plot(tire_groups['TyreLife'], tire_groups['LapTime'], 
            marker='o', linewidth=2, label='Raw Lap Time', color='steelblue')
    
    # Plot fuel-corrected lap time
    if 'LapTime_FuelCorrected' in df.columns:
        ax.plot(tire_groups['TyreLife'], tire_groups['LapTime_FuelCorrected'], 
                marker='s', linewidth=2, label='Fuel-Corrected Lap Time', color='crimson')
    
    ax.set_xlabel('Tire Life (laps)', fontsize=14)
    ax.set_ylabel('Average Lap Time (seconds)', fontsize=14)
    ax.set_title('Tire Degradation: Raw vs Fuel-Corrected', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/f1_tire_degradation_v2.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/f1_tire_degradation_v2.png")
    plt.close()
    
    # Calculate degradation rates
    if 'LapTime_FuelCorrected' in tire_groups.columns:
        raw_slope = np.polyfit(tire_groups['TyreLife'], tire_groups['LapTime'], 1)[0]
        corrected_slope = np.polyfit(tire_groups['TyreLife'], tire_groups['LapTime_FuelCorrected'], 1)[0]
        
        print(f"\nDegradation rates:")
        print(f"  Raw lap time: {raw_slope:+.4f} sec/lap")
        print(f"  Fuel-corrected: {corrected_slope:+.4f} sec/lap")
        
        if raw_slope < 0 and corrected_slope > 0:
            print("\n✓ FUEL MASKING CONFIRMED AND FIXED!")
            print("  V1: Negative slope (fuel burn dominated)")
            print("  V2: Positive slope (tire degradation isolated)")
        elif corrected_slope > 0:
            print("\n✓ Positive degradation rate (physically correct)")

def traffic_effect_analysis(df):
    """Analyze lap time impact of traffic"""
    print("\n" + "=" * 80)
    print("TRAFFIC EFFECT ANALYSIS")
    print("=" * 80)
    
    if 'InTraffic' not in df.columns:
        print("⚠ InTraffic column missing")
        return
    
    # Compare lap times in/out of traffic
    traffic_stats = df.groupby('InTraffic').agg({
        'LapTime': ['mean', 'std', 'count'],
        'LapTime_FuelCorrected': ['mean', 'std'] if 'LapTime_FuelCorrected' in df.columns else ['mean']
    })
    
    print("\nLap time comparison:")
    print(traffic_stats)
    
    # Calculate traffic penalty
    if 'LapTime_FuelCorrected' in df.columns:
        free_air = df[df['InTraffic'] == 0]['LapTime_FuelCorrected'].mean()
        in_traffic = df[df['InTraffic'] == 1]['LapTime_FuelCorrected'].mean()
        penalty = in_traffic - free_air
        
        print(f"\nTraffic penalty (fuel-corrected): {penalty:+.3f} seconds")
        if penalty > 0:
            print("✓ Positive penalty (traffic slows laps)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Box plot comparison
    df_sample = df.sample(min(50000, len(df)), random_state=42)
    df_sample['Traffic'] = df_sample['InTraffic'].map({0: 'Free Air', 1: 'In Traffic'})
    
    if 'LapTime_FuelCorrected' in df_sample.columns:
        df_sample.boxplot(column='LapTime_FuelCorrected', by='Traffic', ax=axes[0])
        axes[0].set_title('Lap Time Distribution by Traffic', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Traffic Status', fontsize=12)
        axes[0].set_ylabel('Fuel-Corrected Lap Time (sec)', fontsize=12)
        plt.sca(axes[0])
        plt.xticks(rotation=0)
    
    # Plot 2: Gap distribution
    if 'GapToCarAhead_sec' in df_sample.columns:
        clean_gaps = df_sample[df_sample['GapToCarAhead_sec'] > 0]['GapToCarAhead_sec']
        clean_gaps_clip = clean_gaps.clip(upper=10)  # Clip at 10 seconds for visibility
        
        axes[1].hist(clean_gaps_clip, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1].axvline(1.5, color='red', linestyle='--', linewidth=2, label='Traffic Threshold (1.5s)')
        axes[1].set_xlabel('Gap to Car Ahead (seconds)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Gap Distribution (Clipped at 10s)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/f1_traffic_analysis_v2.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/f1_traffic_analysis_v2.png")
    plt.close()

def track_comparison(df):
    """Compare different track types"""
    print("\n" + "=" * 80)
    print("TRACK TYPE COMPARISON")
    print("=" * 80)
    
    if 'TrackType' not in df.columns:
        print("⚠ TrackType column missing")
        return
    
    # Average statistics by track type
    track_stats = df.groupby('TrackType').agg({
        'LapTime': 'mean',
        'TrackLength_km': 'mean',
        'TyreLife': 'mean',
        'FuelRemaining_kg': 'mean'
    }).round(2)
    
    print("\nTrack type statistics:")
    print(track_stats)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average lap time by track type
    track_stats['LapTime'].plot(kind='bar', ax=axes[0, 0], color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Average Lap Time by Track Type', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Track Type', fontsize=12)
    axes[0, 0].set_ylabel('Lap Time (seconds)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    plt.sca(axes[0, 0])
    plt.xticks(rotation=0)
    
    # Plot 2: Track length distribution
    track_stats['TrackLength_km'].plot(kind='bar', ax=axes[0, 1], color='darkorange', alpha=0.7)
    axes[0, 1].set_title('Average Track Length by Type', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Track Type', fontsize=12)
    axes[0, 1].set_ylabel('Length (km)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=0)
    
    # Plot 3: Lap time distribution by track type
    df_sample = df.sample(min(50000, len(df)), random_state=42)
    for track_type in df_sample['TrackType'].unique():
        subset = df_sample[df_sample['TrackType'] == track_type]['LapTime']
        axes[1, 0].hist(subset, bins=30, alpha=0.5, label=track_type)
    axes[1, 0].set_xlabel('Lap Time (seconds)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Lap Time Distribution by Track Type', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Track length vs lap time
    if 'TrackLength_km' in df_sample.columns:
        axes[1, 1].scatter(df_sample['TrackLength_km'], df_sample['LapTime'], 
                           c=df_sample['TrackType'].astype('category').cat.codes,
                           cmap='viridis', alpha=0.3, s=1)
        axes[1, 1].set_xlabel('Track Length (km)', fontsize=12)
        axes[1, 1].set_ylabel('Lap Time (seconds)', fontsize=12)
        axes[1, 1].set_title('Track Length vs Lap Time', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/f1_track_comparison_v2.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/f1_track_comparison_v2.png")
    plt.close()

def main():
    """Main execution"""
    # Load data
    df = load_data()
    
    # Basic statistics
    basic_statistics(df)
    
    # V2-specific validations
    fuel_model_validation(df)
    traffic_analysis(df)
    track_type_distribution(df)
    
    # Correlation analysis
    correlation_analysis(df)
    
    # Visualizations
    fuel_effect_visualization(df)
    tire_degradation_analysis(df)
    traffic_effect_analysis(df)
    track_comparison(df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
    print("\nKey files generated:")
    print("  1. f1_correlation_matrix_v2.png")
    print("  2. f1_fuel_analysis_v2.png")
    print("  3. f1_tire_degradation_v2.png")
    print("  4. f1_traffic_analysis_v2.png")
    print("  5. f1_track_comparison_v2.png")
    
    print("\n" + "=" * 80)
    print("V2 VALIDATION SUMMARY")
    print("=" * 80)
    print("\nCheck the outputs above for:")
    print("  ✓ TyreLife ↔ LapTime_FuelCorrected is POSITIVE")
    print("  ✓ FuelRemaining_kg ↔ LapTime is NEGATIVE")
    print("  ✓ NormalizedLap ↔ FuelRemaining_kg ≈ -1.0")
    print("  ✓ InTraffic ↔ LapTime is POSITIVE")
    print("  ✓ TrackLength_km ↔ LapTime is POSITIVE")
    print("\nIf all checks pass, V2 improvements are validated!")
    print("=" * 80)

if __name__ == "__main__":
    main()
