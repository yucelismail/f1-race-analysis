# -*- coding: utf-8 -*-
"""
F1 Data Analysis and Visualization (EDA)
- Correlation matrix heatmap
- Lap time distribution
- Tire degradation analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_FILE = '../data/f1_training_data.csv'
OUTPUT_DIR = '../output/'

def analyze_and_visualize():
    print(f"\n{'='*60}")
    print("F1 Data Analysis (EDA)")
    print(f"{'='*60}")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found!")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Data loaded: {len(df):,} rows")

    # 1. Correlation Matrix
    print("\nCalculating correlation matrix...")
    
    key_columns = [
        'LapTime', 'Position', 'GridPosition', 'TyreLife', 
        'SpeedST', 'TrackTemp', 'AirTemp', 'Rainfall', 
        'GapToLeader', 'Points', 'DriverStandingsPoints'
    ]
    
    available_cols = [c for c in key_columns if c in df.columns]
    corr_matrix = df[available_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('F1 Critical Data - Correlation Matrix')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'f1_correlation_matrix.png')
    print("✓ Correlation matrix saved")

    # 2. Lap Time Distribution
    print("\nGenerating lap time distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['LapTime'] < 120]['LapTime'], bins=50, kde=True, color='blue')
    plt.title('Lap Time Distribution (Normal Race Pace)')
    plt.xlabel('Seconds')
    plt.ylabel('Frequency')
    plt.savefig(OUTPUT_DIR + 'f1_laptime_dist.png')
    print("✓ Lap time distribution saved")

    # 3. Grid vs Position
    print("\nAnalyzing grid vs position relationship...")
    plt.figure(figsize=(10, 6))
    plt.hexbin(df['GridPosition'], df['Position'], gridsize=20, cmap='Blues')
    plt.colorbar(label='Lap Count')
    plt.title('Starting Grid vs Current Position Density')
    plt.xlabel('Grid Position')
    plt.ylabel('Current Position')
    plt.plot([0, 20], [0, 20], 'r--')
    plt.savefig(OUTPUT_DIR + 'f1_grid_vs_pos.png')
    print("✓ Position analysis saved")

    # 4. Tire Degradation
    print("\nAnalyzing tire degradation...")
    if 'TyreLife' in df.columns and 'LapTime' in df.columns:
        plt.figure(figsize=(10, 6))
        dry_laps = df[df['Rainfall'] == 0]
        sample_data = dry_laps.sample(min(10000, len(dry_laps)))
        
        sns.scatterplot(data=sample_data, x='TyreLife', y='LapTime', alpha=0.3)
        sns.regplot(data=sample_data, x='TyreLife', y='LapTime', scatter=False, color='red')
        
        plt.title('Tire Age vs Lap Time (Degradation Effect)')
        plt.ylim(70, 110)
        plt.savefig(OUTPUT_DIR + 'f1_tyre_deg.png')
        print("✓ Tire analysis saved")

    # 5. Export Correlations
    print("\nExporting correlation results...")
    correlations = corr_matrix['LapTime'].sort_values(ascending=False)
    print(correlations)
    
    with open(OUTPUT_DIR + 'correlation_results.txt', 'w', encoding='utf-8') as f:
        f.write(str(correlations))
    print("✓ Results saved to correlation_results.txt")

if __name__ == "__main__":
    analyze_and_visualize()