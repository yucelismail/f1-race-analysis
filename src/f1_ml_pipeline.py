"""
F1 Machine Learning Pipeline - Lap Time Prediction

Goal: Predict F1 lap times using multi-model approach

Models to Compare:
1. Linear Regression (baseline)
2. Random Forest (non-linear, feature importance)
3. XGBoost (gradient boosting, best performance)
4. Neural Network (deep learning, complex patterns)

Target Variable: LapTime (seconds)

Key Features:
- TyreLife, Compound, TyreDistance_km
- FuelRemaining_kg, FuelPenalty_sec
- Circuit characteristics, Weather
- Track status, Traffic indicators
- Sector times (S1, S2, S3)

Strategy:
- Accept negative TyreLife correlation (let models learn non-linear patterns)
- Use all V3.1 features (58 columns)
- Cross-validation for robust evaluation
- Feature importance analysis
- Compound-specific sub-models if needed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'f1_training_data_v3_1.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'ml_models')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class F1MLPipeline:
    """F1 Lap Time Prediction ML Pipeline"""
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        self.feature_names = []
        
    def load_data(self):
        """Load V3.1 dataset"""
        print("=" * 80)
        print("F1 MACHINE LEARNING PIPELINE")
        print("=" * 80)
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Loading data: {INPUT_FILE}")
        self.df = pd.read_csv(INPUT_FILE)
        print(f"  ✓ Loaded {len(self.df):,} laps")
        print(f"  ✓ Features: {len(self.df.columns)} columns\n")
        
        return self
    
    def prepare_features(self):
        """Feature engineering and selection"""
        print("STEP 1: Feature Engineering & Selection")
        print("-" * 80)
        
        df = self.df.copy()
        
        # Target variable
        target = 'LapTime'
        
        # Features to EXCLUDE (not predictive or target-related)
        exclude_cols = [
            target,  # Target variable
            'Year', 'Round',  # Time identifiers (not predictive)
            'LapNumber',  # Already captured in NormalizedLap
            'Driver',  # Too many categories, use team/performance instead
            'CircuitName',  # Already encoded in Circuit
            'CircuitID',  # Redundant with Circuit
            'TrackType',  # Redundant with TrackType_Real
            'TrackLength_km',  # Replaced by TrackLength_Real_km
            'IsAccurate',  # All are 1 after V3 cleaning
            # Sector times - these LEAK future information (known after lap complete)
            'Sector1Time', 'Sector2Time', 'Sector3Time',
            'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
            'LapTime_ms', 'LapTime_FuelCorrected',  # Derived from target
            'LapTime_PerKm', 'LapTime_FuelCorrected_PerKm'  # Also derived
        ]
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values in V3.1 track features
        # For laps without real track data, fill with V3 placeholders
        if 'TrackLength_Real_km' in df.columns:
            df['TrackLength_Real_km'].fillna(4.5, inplace=True)
        if 'TrackTurns' in df.columns:
            df['TrackTurns'].fillna(df['TrackTurns'].median(), inplace=True)
        if 'CornerDensity' in df.columns:
            df['CornerDensity'].fillna(df['CornerDensity'].median(), inplace=True)
        if 'TyreDistance_km' in df.columns:
            df['TyreDistance_km'].fillna(df['TyreLife'] * 4.5, inplace=True)
        
        # Handle categorical variables (encode if not already)
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"  ⚠️  Found {len(categorical_cols)} categorical columns:")
            for col in categorical_cols:
                print(f"     - {col}: {df[col].nunique()} unique values")
                # Drop for now (most should be encoded already)
                feature_cols.remove(col)
        
        # Remove any remaining NaN rows
        df_clean = df[feature_cols + [target]].dropna()
        
        print(f"  ✓ Selected {len(feature_cols)} features")
        print(f"  ✓ Removed {len(df) - len(df_clean):,} rows with missing values")
        print(f"  ✓ Final dataset: {len(df_clean):,} laps\n")
        
        # Show key features
        print(f"  Key Features:")
        important_features = [
            'TyreLife', 'Compound', 'TyreDistance_km',
            'FuelRemaining_kg', 'FuelPenalty_sec',
            'Circuit', 'Weather', 'TrackStatus',
            'TrackLength_Real_km', 'TrackTurns', 'CornerDensity',
            'InTraffic', 'GapToCarAhead_sec',
            'NormalizedLap', 'RaceProgress_pct'
        ]
        for feat in important_features:
            if feat in feature_cols:
                print(f"     ✓ {feat}")
        
        print()
        
        # Store features and target
        X = df_clean[feature_cols]
        y = df_clean[target]
        
        self.feature_names = feature_cols
        
        return X, y
    
    def split_and_scale(self, X, y):
        """Train-test split and feature scaling"""
        print("STEP 2: Train-Test Split & Feature Scaling")
        print("-" * 80)
        
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"  ✓ Train set: {len(X_train):,} laps ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  ✓ Test set: {len(X_test):,} laps ({len(X_test)/len(X)*100:.1f}%)")
        print(f"  ✓ Target variable: {y.name}")
        print(f"     - Mean: {y_train.mean():.2f}s")
        print(f"     - Std: {y_train.std():.2f}s")
        print(f"     - Range: {y_train.min():.2f}s - {y_train.max():.2f}s\n")
        
        # Scale features (important for Neural Networks and regularized models)
        print(f"  Scaling features with StandardScaler...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"  ✓ Features standardized (mean=0, std=1)\n")
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return self
    
    def train_models(self):
        """Train multiple models"""
        print("STEP 3: Train Multiple Models")
        print("-" * 80)
        
        # Model configurations
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.1, random_state=42),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_config['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        # Train each model
        for name, model in models_config.items():
            print(f"\n  Training: {name}")
            print(f"  {'-' * 60}")
            
            start_time = datetime.now()
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predict
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Evaluate
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'training_time': elapsed,
                'y_pred': y_test_pred
            }
            
            print(f"     Train RMSE: {train_rmse:.3f}s | Test RMSE: {test_rmse:.3f}s")
            print(f"     Train MAE:  {train_mae:.3f}s | Test MAE:  {test_mae:.3f}s")
            print(f"     Train R²:   {train_r2:.4f} | Test R²:   {test_r2:.4f}")
            print(f"     Time: {elapsed:.1f}s")
        
        print(f"\n  ✓ All models trained successfully!\n")
        
        return self
    
    def compare_models(self):
        """Compare model performances"""
        print("STEP 4: Model Comparison")
        print("-" * 80)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test RMSE (s)': [r['test_rmse'] for r in self.results.values()],
            'Test MAE (s)': [r['test_mae'] for r in self.results.values()],
            'Test R²': [r['test_r2'] for r in self.results.values()],
            'Train RMSE (s)': [r['train_rmse'] for r in self.results.values()],
            'Overfit Gap': [r['test_rmse'] - r['train_rmse'] for r in self.results.values()],
            'Training Time (s)': [r['training_time'] for r in self.results.values()]
        })
        
        # Sort by Test RMSE (lower is better)
        comparison = comparison.sort_values('Test RMSE (s)')
        
        print(comparison.to_string(index=False))
        print()
        
        # Find best model
        best_model_name = comparison.iloc[0]['Model']
        best_rmse = comparison.iloc[0]['Test RMSE (s)']
        best_r2 = comparison.iloc[0]['Test R²']
        
        print(f"  🏆 Best Model: {best_model_name}")
        print(f"     - Test RMSE: {best_rmse:.3f}s")
        print(f"     - Test R²: {best_r2:.4f}")
        print(f"     - Explains {best_r2*100:.1f}% of lap time variance\n")
        
        return comparison
    
    def feature_importance(self, model_name='Random Forest'):
        """Analyze feature importance"""
        print("STEP 5: Feature Importance Analysis")
        print("-" * 80)
        
        model = self.models.get(model_name)
        
        if model is None:
            print(f"  ⚠️  Model '{model_name}' not found")
            return
        
        # Get feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create DataFrame
            feat_imp = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(f"  Top 15 Most Important Features ({model_name}):")
            print(feat_imp.head(15).to_string(index=False))
            print()
            
            # Plot
            self._plot_feature_importance(feat_imp)
            
        else:
            print(f"  ⚠️  {model_name} doesn't have feature_importances_ attribute")
    
    def _plot_feature_importance(self, feat_imp):
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_n = 20
        feat_imp_top = feat_imp.head(top_n)
        
        ax.barh(range(top_n), feat_imp_top['Importance'])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_imp_top['Feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importances (Random Forest)')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Feature importance plot saved: {output_path}")
        plt.close()
    
    def plot_predictions(self, model_name=None):
        """Plot actual vs predicted lap times"""
        print("\nSTEP 6: Prediction Visualization")
        print("-" * 80)
        
        if model_name is None:
            # Use best model (lowest test RMSE)
            model_name = min(self.results.keys(), 
                           key=lambda k: self.results[k]['test_rmse'])
        
        y_pred = self.results[model_name]['y_pred']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Actual vs Predicted
        axes[0].scatter(self.y_test, y_pred, alpha=0.3, s=1)
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Lap Time (s)')
        axes[0].set_ylabel('Predicted Lap Time (s)')
        axes[0].set_title(f'Actual vs Predicted ({model_name})')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Residuals
        residuals = self.y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.3, s=1)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Lap Time (s)')
        axes[1].set_ylabel('Residual (Actual - Predicted)')
        axes[1].set_title(f'Residual Plot ({model_name})')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'prediction_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Prediction plots saved: {output_path}\n")
        plt.close()
    
    def save_best_model(self):
        """Save best performing model"""
        print("STEP 7: Save Best Model")
        print("-" * 80)
        
        # Find best model
        best_model_name = min(self.results.keys(), 
                             key=lambda k: self.results[k]['test_rmse'])
        best_model = self.models[best_model_name]
        
        # Save model
        model_path = os.path.join(MODELS_DIR, 'f1_best_model.pkl')
        joblib.dump(best_model, model_path)
        print(f"  ✓ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(MODELS_DIR, 'f1_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"  ✓ Scaler saved: {scaler_path}")
        
        # Save feature names
        features_path = os.path.join(MODELS_DIR, 'f1_features.pkl')
        joblib.dump(self.feature_names, features_path)
        print(f"  ✓ Feature names saved: {features_path}")
        
        # Save metadata
        metadata = {
            'model_name': best_model_name,
            'test_rmse': self.results[best_model_name]['test_rmse'],
            'test_mae': self.results[best_model_name]['test_mae'],
            'test_r2': self.results[best_model_name]['test_r2'],
            'n_features': len(self.feature_names),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        metadata_path = os.path.join(MODELS_DIR, 'f1_model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"  ✓ Metadata saved: {metadata_path}\n")
        
        print(f"  🎯 Best Model: {best_model_name}")
        print(f"     Test RMSE: {metadata['test_rmse']:.3f}s")
        print(f"     Test MAE: {metadata['test_mae']:.3f}s")
        print(f"     Test R²: {metadata['test_r2']:.4f}\n")
    
    def run_pipeline(self):
        """Execute full ML pipeline"""
        self.load_data()
        X, y = self.prepare_features()
        self.split_and_scale(X, y)
        self.train_models()
        comparison = self.compare_models()
        self.feature_importance('Random Forest')
        if XGBOOST_AVAILABLE:
            self.feature_importance('XGBoost')
        self.plot_predictions()
        self.save_best_model()
        
        print("=" * 80)
        print("F1 ML PIPELINE COMPLETE")
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        return comparison

def main():
    """Main execution"""
    pipeline = F1MLPipeline()
    comparison = pipeline.run_pipeline()

if __name__ == "__main__":
    main()
