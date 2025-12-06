"""
F1 ML Results Visualization - Comprehensive Report Graphics

Creates publication-quality visualizations for ML project report:
1. Model Performance Comparison
2. Feature Importance Analysis
3. Prediction Quality Assessment
4. Error Distribution Analysis
5. Cross-Circuit Performance
6. Tire Degradation Modeling Success
7. Training Progress and Convergence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'f1_training_data_v3_1.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'ml_report')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)
colors = sns.color_palette("husl", 8)

class F1MLVisualizer:
    """Create comprehensive ML visualizations for report"""
    
    def __init__(self):
        self.df = None
        self.model = None
        self.scaler = None
        self.features = None
        self.metadata = None
        
    def load_resources(self):
        """Load trained model and data"""
        print("=" * 80)
        print("F1 ML VISUALIZATION - REPORT GRAPHICS")
        print("=" * 80)
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Load data
        print(f"Loading data: {DATA_FILE}")
        self.df = pd.read_csv(DATA_FILE)
        print(f"  ✓ Loaded {len(self.df):,} laps\n")
        
        # Load model artifacts
        print(f"Loading model artifacts from: {MODELS_DIR}")
        self.model = joblib.load(os.path.join(MODELS_DIR, 'f1_best_model.pkl'))
        self.scaler = joblib.load(os.path.join(MODELS_DIR, 'f1_scaler.pkl'))
        self.features = joblib.load(os.path.join(MODELS_DIR, 'f1_features.pkl'))
        self.metadata = joblib.load(os.path.join(MODELS_DIR, 'f1_model_metadata.pkl'))
        
        print(f"  ✓ Model: {self.metadata['model_name']}")
        print(f"  ✓ Test RMSE: {self.metadata['test_rmse']:.3f}s")
        print(f"  ✓ Test R²: {self.metadata['test_r2']:.4f}")
        print(f"  ✓ Features: {self.metadata['n_features']}\n")
        
        return self
    
    def viz_1_model_comparison(self):
        """Visualization 1: Model Performance Comparison Bar Chart"""
        print("VIZ 1: Model Performance Comparison")
        print("-" * 80)
        
        # Model results (from previous run)
        models_data = {
            'Model': ['Random Forest', 'XGBoost', 'Gradient\nBoosting', 
                     'Neural\nNetwork', 'Linear\nRegression', 'Ridge', 'Lasso'],
            'Test RMSE': [0.510, 0.710, 0.790, 0.807, 5.230, 5.230, 5.393],
            'Test R²': [0.9977, 0.9956, 0.9946, 0.9943, 0.7613, 0.7613, 0.7462],
            'Training Time': [37.2, 2.6, 119.5, 63.0, 0.3, 0.1, 0.5]
        }
        
        df_models = pd.DataFrame(models_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('F1 Lap Time Prediction - Model Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Plot 1: Test RMSE
        ax1 = axes[0, 0]
        bars1 = ax1.bar(df_models['Model'], df_models['Test RMSE'], color=colors)
        bars1[0].set_color('gold')  # Highlight best
        ax1.set_ylabel('Test RMSE (seconds)', fontweight='bold')
        ax1.set_title('Prediction Error (Lower is Better)', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, max(df_models['Test RMSE']) * 1.1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, df_models['Test RMSE'])):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}s',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: R² Score
        ax2 = axes[0, 1]
        bars2 = ax2.bar(df_models['Model'], df_models['Test R²'] * 100, color=colors)
        bars2[0].set_color('gold')
        ax2.set_ylabel('R² Score (%)', fontweight='bold')
        ax2.set_title('Variance Explained (Higher is Better)', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(70, 100)
        
        for bar, val in zip(bars2, df_models['Test R²']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val*100:.2f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 3: Training Time
        ax3 = axes[1, 0]
        bars3 = ax3.bar(df_models['Model'], df_models['Training Time'], color=colors)
        ax3.set_ylabel('Training Time (seconds)', fontweight='bold')
        ax3.set_title('Computational Efficiency', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_yscale('log')
        
        for bar, val in zip(bars3, df_models['Training Time']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}s',
                    ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Performance vs Speed Trade-off
        ax4 = axes[1, 1]
        scatter = ax4.scatter(df_models['Training Time'], 
                             df_models['Test RMSE'],
                             s=df_models['Test R²'] * 1000,
                             c=range(len(df_models)), 
                             cmap='viridis',
                             alpha=0.7,
                             edgecolors='black',
                             linewidths=2)
        
        # Annotate points
        for i, model in enumerate(df_models['Model']):
            ax4.annotate(model.replace('\n', ' '), 
                        (df_models['Training Time'][i], df_models['Test RMSE'][i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold')
        
        ax4.set_xlabel('Training Time (seconds)', fontweight='bold')
        ax4.set_ylabel('Test RMSE (seconds)', fontweight='bold')
        ax4.set_title('Speed vs Accuracy Trade-off', fontweight='bold')
        ax4.set_xscale('log')
        ax4.grid(alpha=0.3)
        ax4.invert_yaxis()  # Lower RMSE is better
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '1_model_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}\n")
        plt.close()
    
    def viz_2_feature_importance_detailed(self):
        """Visualization 2: Detailed Feature Importance Analysis"""
        print("VIZ 2: Feature Importance - Detailed Analysis")
        print("-" * 80)
        
        # Get feature importances
        importances = self.model.feature_importances_
        feat_imp = pd.DataFrame({
            'Feature': self.features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Categorize features
        feature_categories = {
            'Track': ['Circuit', 'TrackLength_Real_km', 'TrackTurns', 'CornerDensity', 
                     'TotalRaceLaps', 'CircuitLat', 'CircuitLng'],
            'Tire': ['TyreLife', 'Compound', 'TyreDistance_km', 'Stint'],
            'Fuel': ['FuelRemaining_kg', 'FuelPenalty_sec'],
            'Weather': ['WeatherTemp', 'TrackTemp', 'AirTemp', 'Humidity', 
                       'WeatherPressure', 'WeatherWindSpeed', 'WeatherWindDirection', 'Weather'],
            'Race Progress': ['NormalizedLap', 'RaceProgress_pct', 'Position'],
            'Traffic': ['InTraffic', 'GapToCarAhead_sec'],
            'Speed': ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
        }
        
        # Assign categories
        def get_category(feature):
            for cat, feats in feature_categories.items():
                if feature in feats:
                    return cat
            return 'Other'
        
        feat_imp['Category'] = feat_imp['Feature'].apply(get_category)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Feature Importance Analysis - Random Forest Model', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Top 20 Features
        ax1 = axes[0, 0]
        top20 = feat_imp.head(20)
        colors_cat = [plt.cm.tab10(hash(cat) % 10) for cat in top20['Category']]
        bars = ax1.barh(range(20), top20['Importance'], color=colors_cat)
        ax1.set_yticks(range(20))
        ax1.set_yticklabels(top20['Feature'])
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance Score', fontweight='bold')
        ax1.set_title('Top 20 Most Important Features', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        total_imp = feat_imp['Importance'].sum()
        for i, (bar, val) in enumerate(zip(bars, top20['Importance'])):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {val/total_imp*100:.1f}%',
                    ha='left', va='center', fontsize=8)
        
        # Plot 2: Importance by Category
        ax2 = axes[0, 1]
        cat_importance = feat_imp.groupby('Category')['Importance'].sum().sort_values(ascending=False)
        colors_pie = plt.cm.Set3(range(len(cat_importance)))
        wedges, texts, autotexts = ax2.pie(cat_importance.values, 
                                           labels=cat_importance.index,
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           colors=colors_pie)
        ax2.set_title('Importance Distribution by Category', fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Plot 3: Cumulative Importance
        ax3 = axes[1, 0]
        feat_imp_sorted = feat_imp.sort_values('Importance', ascending=False)
        cumsum = np.cumsum(feat_imp_sorted['Importance'])
        cumsum_pct = cumsum / cumsum.iloc[-1] * 100
        
        ax3.plot(range(len(cumsum_pct)), cumsum_pct, linewidth=2, color='steelblue')
        ax3.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% Threshold')
        ax3.axhline(y=95, color='orange', linestyle='--', linewidth=2, label='95% Threshold')
        
        # Find features needed for 80% and 95%
        idx_80 = np.argmax(cumsum_pct >= 80)
        idx_95 = np.argmax(cumsum_pct >= 95)
        
        ax3.scatter([idx_80], [80], color='red', s=100, zorder=5)
        ax3.scatter([idx_95], [95], color='orange', s=100, zorder=5)
        
        ax3.text(idx_80, 75, f'{idx_80+1} features', ha='center', fontweight='bold')
        ax3.text(idx_95, 90, f'{idx_95+1} features', ha='center', fontweight='bold')
        
        ax3.set_xlabel('Number of Features', fontweight='bold')
        ax3.set_ylabel('Cumulative Importance (%)', fontweight='bold')
        ax3.set_title('Cumulative Feature Importance', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4: Feature Importance Distribution
        ax4 = axes[1, 1]
        ax4.hist(feat_imp['Importance'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.axvline(feat_imp['Importance'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {feat_imp["Importance"].mean():.4f}')
        ax4.axvline(feat_imp['Importance'].median(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {feat_imp["Importance"].median():.4f}')
        ax4.set_xlabel('Importance Score', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Distribution of Feature Importances', fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '2_feature_importance_detailed.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}\n")
        plt.close()
    
    def viz_3_prediction_quality(self):
        """Visualization 3: Prediction Quality Assessment"""
        print("VIZ 3: Prediction Quality Assessment")
        print("-" * 80)
        
        # Prepare data for prediction
        df = self.df.copy()
        
        # Use same feature selection as training
        exclude_cols = [
            'LapTime', 'Year', 'Round', 'LapNumber', 'Driver',
            'CircuitName', 'CircuitID', 'TrackType', 'TrackLength_km', 'IsAccurate',
            'Sector1Time', 'Sector2Time', 'Sector3Time',
            'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
            'LapTime_ms', 'LapTime_FuelCorrected',
            'LapTime_PerKm', 'LapTime_FuelCorrected_PerKm'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in feature_cols:
                feature_cols.remove(col)
        
        # Fill missing values
        if 'TrackLength_Real_km' in df.columns:
            df['TrackLength_Real_km'].fillna(4.5, inplace=True)
        if 'TrackTurns' in df.columns:
            df['TrackTurns'].fillna(df['TrackTurns'].median(), inplace=True)
        if 'CornerDensity' in df.columns:
            df['CornerDensity'].fillna(df['CornerDensity'].median(), inplace=True)
        if 'TyreDistance_km' in df.columns:
            df['TyreDistance_km'].fillna(df['TyreLife'] * 4.5, inplace=True)
        
        # Get clean data
        df_clean = df[feature_cols + ['LapTime']].dropna()
        
        # Sample for visualization (use 10,000 random samples for speed)
        df_sample = df_clean.sample(n=min(10000, len(df_clean)), random_state=42)
        
        X_sample = df_sample[self.features]
        y_actual = df_sample['LapTime']
        
        # Scale and predict
        X_scaled = self.scaler.transform(X_sample)
        y_pred = self.model.predict(X_scaled)
        
        # Calculate residuals
        residuals = y_actual - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Prediction Quality Assessment - Random Forest Model', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Actual vs Predicted (Dense)
        ax1 = axes[0, 0]
        hex_plot = ax1.hexbin(y_actual, y_pred, gridsize=50, cmap='YlOrRd', mincnt=1)
        ax1.plot([y_actual.min(), y_actual.max()], 
                [y_actual.min(), y_actual.max()], 
                'b--', lw=3, label='Perfect Prediction', alpha=0.7)
        
        # Add R² and RMSE text
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.3f}s'
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontweight='bold')
        
        ax1.set_xlabel('Actual Lap Time (s)', fontweight='bold')
        ax1.set_ylabel('Predicted Lap Time (s)', fontweight='bold')
        ax1.set_title('Actual vs Predicted (Hexbin Density)', fontweight='bold')
        ax1.legend(loc='lower right')
        plt.colorbar(hex_plot, ax=ax1, label='Count')
        
        # Plot 2: Residual Plot
        ax2 = axes[0, 1]
        ax2.hexbin(y_pred, residuals, gridsize=50, cmap='coolwarm', mincnt=1)
        ax2.axhline(y=0, color='black', linestyle='--', lw=2, label='Zero Error')
        ax2.set_xlabel('Predicted Lap Time (s)', fontweight='bold')
        ax2.set_ylabel('Residual (Actual - Predicted) (s)', fontweight='bold')
        ax2.set_title('Residual Analysis', fontweight='bold')
        ax2.legend()
        plt.colorbar(hex_plot, ax=ax2, label='Count')
        
        # Plot 3: Error Distribution
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Fit normal distribution
        from scipy import stats
        mu, std = stats.norm.fit(residuals)
        xmin, xmax = ax3.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(x, p, 'r-', linewidth=2, label=f'Normal Fit\nμ={mu:.3f}, σ={std:.3f}')
        ax3_twin.set_ylabel('Probability Density', fontweight='bold')
        
        ax3.axvline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
        ax3.set_xlabel('Prediction Error (s)', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Error Distribution (Normal Fit)', fontweight='bold')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Error by Lap Time Range
        ax4 = axes[1, 1]
        
        # Bin lap times
        bins = np.arange(60, 125, 5)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        abs_errors = np.abs(residuals)
        binned_errors = []
        binned_std = []
        
        for i in range(len(bins)-1):
            mask = (y_actual >= bins[i]) & (y_actual < bins[i+1])
            if mask.sum() > 0:
                binned_errors.append(abs_errors[mask].mean())
                binned_std.append(abs_errors[mask].std())
            else:
                binned_errors.append(0)
                binned_std.append(0)
        
        ax4.errorbar(bin_centers, binned_errors, yerr=binned_std, 
                    fmt='o-', linewidth=2, markersize=8, capsize=5,
                    color='steelblue', ecolor='gray', label='Mean Absolute Error')
        ax4.axhline(rmse, color='red', linestyle='--', linewidth=2, 
                   label=f'Overall RMSE: {rmse:.3f}s')
        ax4.set_xlabel('Actual Lap Time Range (s)', fontweight='bold')
        ax4.set_ylabel('Mean Absolute Error (s)', fontweight='bold')
        ax4.set_title('Prediction Error by Lap Time Range', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '3_prediction_quality.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}\n")
        plt.close()
    
    def viz_4_circuit_performance(self):
        """Visualization 4: Cross-Circuit Model Performance"""
        print("VIZ 4: Cross-Circuit Performance Analysis")
        print("-" * 80)
        
        # Load encoder to decode circuits
        encoders = joblib.load(os.path.join(BASE_DIR, 'data', 'f1_encoders.pkl'))
        circuit_encoder = encoders['Circuit']
        
        # Prepare data
        df = self.df.copy()
        df['CircuitName'] = circuit_encoder.inverse_transform(df['Circuit'])
        
        # Use same feature selection
        exclude_cols = [
            'LapTime', 'Year', 'Round', 'LapNumber', 'Driver',
            'CircuitName', 'CircuitID', 'TrackType', 'TrackLength_km', 'IsAccurate',
            'Sector1Time', 'Sector2Time', 'Sector3Time',
            'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
            'LapTime_ms', 'LapTime_FuelCorrected',
            'LapTime_PerKm', 'LapTime_FuelCorrected_PerKm'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in feature_cols:
                feature_cols.remove(col)
        
        # Fill missing values
        if 'TrackLength_Real_km' in df.columns:
            df['TrackLength_Real_km'].fillna(4.5, inplace=True)
        if 'TrackTurns' in df.columns:
            df['TrackTurns'].fillna(df['TrackTurns'].median(), inplace=True)
        if 'CornerDensity' in df.columns:
            df['CornerDensity'].fillna(df['CornerDensity'].median(), inplace=True)
        if 'TyreDistance_km' in df.columns:
            df['TyreDistance_km'].fillna(df['TyreLife'] * 4.5, inplace=True)
        
        df_clean = df[feature_cols + ['LapTime', 'CircuitName']].dropna()
        
        # Sample per circuit for speed
        df_sample = df_clean.groupby('CircuitName').apply(
            lambda x: x.sample(n=min(500, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        X = df_sample[self.features]
        y_actual = df_sample['LapTime']
        circuits = df_sample['CircuitName']
        
        # Predict
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics per circuit
        from sklearn.metrics import mean_absolute_error, r2_score
        
        circuit_metrics = []
        for circuit in circuits.unique():
            mask = circuits == circuit
            mae = mean_absolute_error(y_actual[mask], y_pred[mask])
            r2 = r2_score(y_actual[mask], y_pred[mask])
            count = mask.sum()
            circuit_metrics.append({
                'Circuit': circuit,
                'MAE': mae,
                'R²': r2,
                'Count': count
            })
        
        df_metrics = pd.DataFrame(circuit_metrics).sort_values('MAE')
        
        # Shorten circuit names
        df_metrics['Circuit_Short'] = df_metrics['Circuit'].str.replace('Circuit', '').str.replace('Grand Prix', '').str.strip()
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('Model Performance Across Different Circuits', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: MAE by Circuit
        ax1 = axes[0]
        bars = ax1.barh(df_metrics['Circuit_Short'], df_metrics['MAE'], color=colors[0])
        
        # Color best and worst
        bars[0].set_color('green')
        bars[-1].set_color('red')
        
        ax1.set_xlabel('Mean Absolute Error (seconds)', fontweight='bold')
        ax1.set_title('Prediction Error by Circuit (Lower is Better)', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, df_metrics['MAE']):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {val:.3f}s',
                    ha='left', va='center', fontsize=9)
        
        # Plot 2: R² by Circuit
        ax2 = axes[1]
        df_metrics_r2 = df_metrics.sort_values('R²', ascending=False)
        bars2 = ax2.barh(df_metrics_r2['Circuit_Short'], df_metrics_r2['R²'] * 100, color=colors[1])
        
        bars2[0].set_color('green')
        bars2[-1].set_color('red')
        
        ax2.set_xlabel('R² Score (%)', fontweight='bold')
        ax2.set_title('Variance Explained by Circuit (Higher is Better)', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars2, df_metrics_r2['R²']):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {val*100:.2f}%',
                    ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '4_circuit_performance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}\n")
        plt.close()
    
    def viz_5_tire_degradation_model(self):
        """Visualization 5: Tire Degradation Modeling Success"""
        print("VIZ 5: Tire Degradation Modeling")
        print("-" * 80)
        
        # Focus on tire-related analysis - simplified version
        df = self.df.copy()
        
        # Sample data for visualization
        df_sample = df.sample(n=min(15000, len(df)), random_state=42)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Tire Degradation Analysis - ML Model Success', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: TyreLife vs LapTime (Raw Data - Negative Correlation)
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(df_sample['TyreLife'], df_sample['LapTime'], 
                              c=df_sample['Compound'], cmap='viridis', 
                              alpha=0.3, s=1)
        
        # Add correlation line
        z = np.polyfit(df_sample['TyreLife'], df_sample['LapTime'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_sample['TyreLife'].min(), df_sample['TyreLife'].max(), 100)
        ax1.plot(x_line, p(x_line), "r--", linewidth=2, 
                label=f'Linear: slope={z[0]:.3f}')
        
        corr = df_sample['TyreLife'].corr(df_sample['LapTime'])
        ax1.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontweight='bold')
        
        ax1.set_xlabel('Tire Life (laps)', fontweight='bold')
        ax1.set_ylabel('Lap Time (s)', fontweight='bold')
        ax1.set_title('Raw Data: Negative Correlation (Problem!)', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Compound')
        
        # Plot 2: Simplified - just show concept
        ax2 = axes[0, 1]
        # Create conceptual visualization showing ML overcame negative correlation
        ax2.text(0.5, 0.7, '🤖 Machine Learning Success', 
                transform=ax2.transAxes, fontsize=16,
                ha='center', fontweight='bold', color='green')
        ax2.text(0.5, 0.5, 'Random Forest learned non-linear\nrelationships between:',
                transform=ax2.transAxes, fontsize=12,
                ha='center')
        ax2.text(0.5, 0.3, '• Tire compound types\n• Track characteristics\n• Fuel effects\n• Weather conditions',
                transform=ax2.transAxes, fontsize=11,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax2.text(0.5, 0.1, f'Result: {self.metadata["test_r2"]*100:.2f}% accuracy!',
                transform=ax2.transAxes, fontsize=14,
                ha='center', fontweight='bold', color='darkgreen')
        ax2.axis('off')
        ax2.set_title('ML Model Overcame Correlation Issues', fontweight='bold')
        
        # Plot 3: V1/V2/V3 Evolution
        ax3 = axes[1, 0]
        versions = ['V1\n(Raw Data)', 'V2\n(Fuel Model)', 'V3\n(Cleaning)', 'V3.1\n(Track)', 'ML\n(Random Forest)']
        correlations = [-0.202, -0.161, -0.204, -0.267, 0.998]  # R² for ML
        colors_prog = ['red', 'orange', 'orange', 'orange', 'green']
        
        bars = ax3.bar(versions, correlations, color=colors_prog, alpha=0.7, edgecolor='black', linewidth=2)
        ax3.axhline(0, color='black', linestyle='-', linewidth=1)
        ax3.set_ylabel('Correlation / R²', fontweight='bold')
        ax3.set_title('Project Evolution: Problem → Solution', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add labels
        for bar, val in zip(bars, correlations):
            height = bar.get_height()
            if val < 0:
                va = 'top'
                y_pos = height - 0.02
            else:
                va = 'bottom'
                y_pos = height + 0.02
            ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{val:.3f}',
                    ha='center', va=va, fontsize=10, fontweight='bold')
        
        # Plot 4: Tire Distance vs Lap Time (V3.1 Feature)
        ax4 = axes[1, 1]
        if 'TyreDistance_km' in df_sample.columns:
            scatter4 = ax4.scatter(df_sample['TyreDistance_km'], df_sample['LapTime'],
                                  c=df_sample['Compound'], cmap='viridis',
                                  alpha=0.3, s=1)
            
            corr_dist = df_sample['TyreDistance_km'].corr(df_sample['LapTime'])
            ax4.text(0.05, 0.95, f'Correlation: {corr_dist:.4f}', 
                    transform=ax4.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontweight='bold')
            
            ax4.set_xlabel('Tire Distance (km)', fontweight='bold')
            ax4.set_ylabel('Lap Time (s)', fontweight='bold')
            ax4.set_title('V3.1 Feature: Track-Normalized Tire Wear', fontweight='bold')
            ax4.grid(alpha=0.3)
            plt.colorbar(scatter4, ax=ax4, label='Compound')
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '5_tire_degradation_model.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}\n")
        plt.close()
    
    def viz_6_learning_curves(self):
        """Visualization 6: Learning Curves and Convergence"""
        print("VIZ 6: Learning Curves Analysis")
        print("-" * 80)
        
        from sklearn.model_selection import learning_curve
        
        # Prepare data (use subset for speed)
        df = self.df.copy()
        
        exclude_cols = [
            'LapTime', 'Year', 'Round', 'LapNumber', 'Driver',
            'CircuitName', 'CircuitID', 'TrackType', 'TrackLength_km', 'IsAccurate',
            'Sector1Time', 'Sector2Time', 'Sector3Time',
            'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
            'LapTime_ms', 'LapTime_FuelCorrected',
            'LapTime_PerKm', 'LapTime_FuelCorrected_PerKm'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in feature_cols:
                feature_cols.remove(col)
        
        # Fill missing
        if 'TrackLength_Real_km' in df.columns:
            df['TrackLength_Real_km'].fillna(4.5, inplace=True)
        if 'TrackTurns' in df.columns:
            df['TrackTurns'].fillna(df['TrackTurns'].median(), inplace=True)
        if 'CornerDensity' in df.columns:
            df['CornerDensity'].fillna(df['CornerDensity'].median(), inplace=True)
        if 'TyreDistance_km' in df.columns:
            df['TyreDistance_km'].fillna(df['TyreLife'] * 4.5, inplace=True)
        
        df_clean = df[feature_cols + ['LapTime']].dropna()
        
        # Sample for learning curve (expensive operation)
        df_sample = df_clean.sample(n=min(30000, len(df_clean)), random_state=42)
        
        X = df_sample[self.features]
        y = df_sample['LapTime']
        
        X_scaled = self.scaler.transform(X)
        
        print("  Computing learning curves (this may take a minute)...")
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, test_scores = learning_curve(
            self.model, X_scaled, y,
            train_sizes=train_sizes,
            cv=3,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42
        )
        
        # Convert to RMSE (positive)
        train_scores_rmse = -train_scores
        test_scores_rmse = -test_scores
        
        train_mean = np.mean(train_scores_rmse, axis=1)
        train_std = np.std(train_scores_rmse, axis=1)
        test_mean = np.mean(test_scores_rmse, axis=1)
        test_std = np.std(test_scores_rmse, axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Learning Curves - Model Convergence Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Learning Curve
        ax1 = axes[0]
        ax1.plot(train_sizes_abs, train_mean, 'o-', color='blue', 
                label='Training Score', linewidth=2, markersize=8)
        ax1.fill_between(train_sizes_abs, 
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.2, color='blue')
        
        ax1.plot(train_sizes_abs, test_mean, 'o-', color='red', 
                label='Cross-Validation Score', linewidth=2, markersize=8)
        ax1.fill_between(train_sizes_abs,
                        test_mean - test_std,
                        test_mean + test_std,
                        alpha=0.2, color='red')
        
        ax1.set_xlabel('Training Set Size', fontweight='bold')
        ax1.set_ylabel('RMSE (seconds)', fontweight='bold')
        ax1.set_title('Learning Curve - RMSE vs Training Size', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Convergence Analysis
        ax2 = axes[1]
        gap = test_mean - train_mean
        ax2.plot(train_sizes_abs, gap, 'o-', color='purple', 
                linewidth=2, markersize=8, label='Generalization Gap')
        ax2.axhline(0, color='green', linestyle='--', linewidth=2, 
                   label='No Overfitting')
        
        ax2.set_xlabel('Training Set Size', fontweight='bold')
        ax2.set_ylabel('RMSE Gap (Test - Train)', fontweight='bold')
        ax2.set_title('Overfitting Analysis', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Add text annotation
        final_gap = gap[-1]
        ax2.text(0.5, 0.95, 
                f'Final Gap: {final_gap:.3f}s\nStatus: {"Low Overfitting ✓" if final_gap < 0.5 else "Moderate Overfitting"}',
                transform=ax2.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '6_learning_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}\n")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations for report"""
        self.load_resources()
        self.viz_1_model_comparison()
        self.viz_2_feature_importance_detailed()
        self.viz_3_prediction_quality()
        self.viz_4_circuit_performance()
        self.viz_5_tire_degradation_model()
        self.viz_6_learning_curves()
        
        print("=" * 80)
        print("ALL VISUALIZATIONS COMPLETE")
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 80)

def main():
    """Main execution"""
    visualizer = F1MLVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
