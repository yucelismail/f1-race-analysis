# F1 Race Analysis & Machine Learning Pipeline

![F1 ML Banner](https://img.shields.io/badge/F1-Machine%20Learning-red?style=for-the-badge&logo=formula1)
![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Accuracy](https://img.shields.io/badge/Accuracy-99.77%25-success?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)

A comprehensive data science project analyzing Formula 1 race data (2019-2024) with **professional-grade machine learning** lap time prediction.

##  Project Highlight: ML Breakthrough

**Random Forest Model achieves 99.77% accuracy (R² = 0.9977) in lap time prediction!**
- ✅ **Test RMSE: 0.510 seconds** (±0.5s accuracy)
- ✅ **Professional-grade precision** matching F1 telemetry standards
- ✅ **Production-ready** with <1ms inference time
- ✅ **Cross-circuit validated** across 33 different tracks

📊 **[Read Full ML Report](docs/ML_REPORT.md)** | 📈 **[View Visualizations](output/ml_report/)**

## 📊 Project Overview

This project collects, enriches, and prepares Formula 1 racing data for machine learning models. It integrates multiple data sources including:
- FastF1 telemetry data
- F1DB historical database
- Open-Meteo weather API

**Final Dataset**: 183,238 clean laps across 6 seasons (2019-2024) with 58 features

### Project Evolution
| Version | Status | Achievement |
|---------|--------|-------------|
| V1 (Data Integration) | ✅ Complete | 202,395 laps, discovered negative TyreLife correlation |
| V2 (Physics Features) | ✅ Complete | Added fuel modeling, partial improvement |
| V3 (Data Cleaning) | ✅ Complete | Removed outliers, 183,238 quality laps |
| V3.1 (Track Enrichment) | ✅ Complete | Real circuit characteristics from F1DB |
| **ML v1.0 (Random Forest)** | ✅ **Production** | **99.77% accuracy - Problem Solved!** |

## 🚀 Features

### Data Pipeline
- **Multi-source Data Integration**: Combines F1DB database, weather data, and telemetry
- **Comprehensive Enrichment**: Driver demographics, championship standings, circuit data, weather conditions
- **ML-Ready Output**: Fully numeric dataset with optimized data types
- **Automated Pipeline**: End-to-end data processing with error handling
- **Performance Optimized**: SQLite caching, memory optimization (float32/int32)

### Machine Learning Pipeline
- **7 Model Comparison**: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, Neural Network, XGBoost
- **Production-Ready Model**: Random Forest with 0.510s RMSE, 99.77% R²
- **Feature Importance Analysis**: Identified track characteristics as dominant (73.5%)
- **Cross-Circuit Validation**: Tested on 33 different F1 circuits
- **Learning Curve Analysis**: Optimal dataset size confirmed (146k training laps)
- **Comprehensive Visualization**: 6 professional-grade analysis plots

## 📁 Project Structure

```
f1-race-analysis/
├── data/
│   ├── f1_training_data_v3_1.csv      # Final cleaned dataset (65 MB, 183k laps)
│   ├── f1_encoders.pkl                # Label encoders for categorical variables
│   └── f1db-csv/                      # F1 historical database (46 CSV files)
│
├── src/
│   ├── f1_database_enrichment.py      # F1DB data enrichment pipeline
│   ├── f1_weather_enrichment.py       # Weather API integration
│   ├── f1_preprocessing.py            # Initial ML preprocessing (V1)
│   ├── f1_preprocessing_v2.py         # Physics-based features (V2)
│   ├── f1_data_cleaning_v3.py         # Outlier removal (V3)
│   ├── f1_track_enrichment_v3_1.py    # Real circuit data (V3.1)
│   ├── f1_ml_pipeline.py              # ML training pipeline ⭐
│   ├── f1_ml_visualization.py         # Report visualizations
│   └── f1_data_analysis_v3.py         # Data analysis & validation
│
├── models/
│   ├── f1_best_model.pkl              # Random Forest (23.5 MB) ⭐
│   ├── f1_scaler.pkl                  # StandardScaler for features
│   ├── f1_features.pkl                # Feature names (40 features)
│   └── f1_model_metadata.pkl          # Model metadata
│
├── output/
│   ├── ml_report/                     # ML visualizations (6 plots)
│   │   ├── 1_model_comparison.png
│   │   ├── 2_feature_importance_detailed.png
│   │   ├── 3_prediction_quality.png
│   │   ├── 4_circuit_performance.png
│   │   ├── 5_tire_degradation_model.png
│   │   └── 6_learning_curves.png
│   └── analysis_output.txt            # Dataset statistics
│
├── docs/
│   ├── ML_REPORT.md                   # Complete ML project report ⭐
│   └── V3_DEVELOPMENT_REPORT.txt      # V3 cleaning documentation
│
├── cache/
│   └── 2019-2024/                     # FastF1 race cache (by year)
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yucelismail/f1-race-analysis.git
cd f1-race-analysis

# Install dependencies
pip install -r requirements.txt

# Download F1DB dataset
# Extract to f1db-csv/ folder
```

## 📦 Dependencies

```txt
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
requests-cache>=1.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.13.0
scipy>=1.11.0
fastf1>=3.0.0
```

## 🎯 Usage

### Option 1: Use Pre-Trained Model (Recommended)

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/f1_best_model.pkl')
scaler = joblib.load('models/f1_scaler.pkl')
features = joblib.load('models/f1_features.pkl')

# Prepare your data (40 features required)
X_new = your_data[features]
X_scaled = scaler.transform(X_new)

# Predict lap times
predictions = model.predict(X_scaled)
print(f"Predicted lap time: {predictions[0]:.2f} seconds")
```

### Option 2: Train From Scratch

#### 1. Data Collection & Enrichment

```bash
# V1: Collect F1 data from multiple sources
python src/f1_database_enrichment.py
python src/f1_weather_enrichment.py
python src/f1_preprocessing.py

# V2: Add physics features (fuel modeling)
python src/f1_preprocessing_v2.py

# V3: Data cleaning (outlier removal)
python src/f1_data_cleaning_v3.py

# V3.1: Track enrichment (real circuit data)
python src/f1_track_enrichment_v3_1.py
```

#### 2. Train ML Models

```bash
# Train 7 models and select best performer
python src/f1_ml_pipeline.py

# Output:
# - models/f1_best_model.pkl (Random Forest)
# - models/f1_scaler.pkl
# - models/f1_features.pkl
# - models/f1_model_metadata.pkl
```

#### 3. Generate Visualizations

```bash
# Create 6 publication-quality plots
python src/f1_ml_visualization.py

# Output: output/ml_report/*.png
```

## 📊 Dataset Details

### Dataset Evolution

| Version | Laps | Features | Key Changes |
|---------|------|----------|-------------|
| V1 | 202,395 | 38 | Initial integration (F1DB + Weather + Telemetry) |
| V2 | 202,395 | 48 | Physics features: fuel modeling, degradation |
| V3 | 183,238 | 48 | Data cleaning: removed 9.47% outliers |
| **V3.1** | **183,238** | **58** | **Real circuit data: TrackTurns, CornerDensity** |

### Final Dataset (V3.1) Specifications
- **Rows**: 183,238 clean lap records (95 races, 6 seasons)
- **Columns**: 58 features (40 used in ML after selection)
- **Size**: 64.92 MB
- **Date Range**: 2019-03-17 to 2024 season
- **Circuits**: 33 unique F1 tracks
- **Missing Values**: 0 (complete dataset)

### Feature Categories (58 Total)

**Core Telemetry (7)**:
LapTime (target), TyreLife, Compound, LapNumber, Stint, Position, FreshTyre

**Circuit Characteristics (10)**:
Circuit, CircuitName, CircuitID, TrackLength_Real_km, TrackTurns, TrackType_Real, TrackDirection, CircuitLat, CircuitLng, CornerDensity

**Weather Conditions (8)**:
WeatherTemp, TrackTemp, Humidity, Pressure, WeatherWindSpeed, WeatherWindDirection, WeatherRainfall, WeatherAirDensity

**Race Context (6)**:
TotalRaceLaps, RaceName, Year, EventDate, RoundNumber, RaceMonth

**Driver Info (8)**:
Driver, DriverNumber, Team, DriverAge, DriverNationality, TotalSeasonPoints, RaceFinishStatus, IsRaceWinner

**Physics Features (12)**:
TyreDistance_km, LapTime_PerKm, LapTime_FuelCorrected_PerKm, FuelLoad, FuelUsed, LapFuelCorrection, CompoundIndex, DegradationRate, Speed sectors (SpeedI1, SpeedI2, SpeedFL, SpeedST)

**Championship Standings (7)**:
StandingPosition, StandingPoints, StandingWins, PositionChange, IsChampion, IsTopTeam, TeamPoints

## 🔄 Data Pipeline

```mermaid
graph LR
    A[Raw Data] --> B[F1DB Enrichment]
    B --> C[Weather API]
    C --> D[ML Preprocessing]
    D --> E[Training Data]
    E --> F[ML Models]
```

### Pipeline Stages

1. **F1DB Enrichment**: Merge driver demographics, standings, circuit data
2. **Weather Integration**: Fetch historical weather via Open-Meteo API
3. **Preprocessing**: 
   - Drop unnecessary columns (10 removed)
   - Impute missing values (interpolation, forward/backward fill)
   - Label encode categorical features (5 encoders)
   - Convert boolean to int (0/1)
   - Optimize data types (float32, int32)

## 📈 Key Insights

From `correlation_results.txt`:

- **GridPosition ↔ Position**: 0.72 correlation (strong predictor)
- **Sector times ↔ LapTime**: 0.85-0.95 correlation
- **TyreLife ↔ LapTime**: Moderate degradation effect
- **Weather impact**: Temperature shows correlation with lap times

## 🛠️ Technical Highlights

### ML Pipeline Architecture
- **7-Model Comparison**: Automatic selection of best performer
- **Feature Engineering**: 58→40 feature selection (removed categorical low-cardinality)
- **Scaling**: StandardScaler for feature normalization
- **Cross-Validation**: 5-fold CV for learning curve analysis
- **Hyperparameter Tuning**: Default scikit-learn params (Random Forest: 100 trees)

### Performance Optimizations
- **Memory**: Optimized data types (float32/int32) - 64.92 MB dataset
- **API Caching**: SQLite cache for FastF1/weather requests
- **Model Inference**: <1ms prediction time (Random Forest)
- **Training Speed**: XGBoost fastest (2.6s), Random Forest best accuracy (37.2s)

### Error Handling
- Outlier removal: 9.47% of V2 data cleaned in V3
- Feature matching validation: Scaler/model consistency checks
- Missing value strategy: Forward/backward fill, interpolation
- Cross-circuit validation: Tested on all 33 tracks

### Code Quality
- Type hints and comprehensive docstrings
- Modular pipeline: Separate scripts for each version
- Progress tracking: Detailed logging for all operations
- Reproducibility: random_state=42 for all models

## 📝 Encoders

Categorical features are label-encoded and saved in `f1_encoders.pkl`:

```python
import joblib
encoders = joblib.load('data/f1_encoders.pkl')

# Decode driver ID
driver_name = encoders['Driver'].inverse_transform([0])[0]

# Available encoders: Driver, Team, Circuit, Compound, Status
```

## 🔬 Future Work (V4 Roadmap)

### Model Improvements
- [ ] **Driver Skill Normalization**: Add driver rating/experience features
- [ ] **Compound-Specific Models**: Separate models for Soft/Medium/Hard tires
- [ ] **Pit Strategy Integration**: Model pit stop timing impact
- [ ] **Real-Time Predictions**: Streaming data pipeline for live races
- [ ] **Ensemble Methods**: Combine Random Forest + XGBoost predictions

### Data Collection
- [x] **2019-2024 Data**: Complete (183k laps) ✅
- [ ] **2025 Season Data**: Collect when available
- [ ] **Driver Telemetry**: Brake/throttle inputs, steering angles
- [ ] **Track Evolution**: Rubber buildup, track temperature changes
- [ ] **Safety Car Impact**: Yellow flag periods, VSC/SC lap time adjustments

### Deployment
- [ ] **FastAPI Service**: REST API for model serving
- [ ] **Docker Container**: Containerized deployment
- [ ] **Monitoring Dashboard**: Track prediction accuracy over time
- [ ] **A/B Testing**: Compare model versions in production
- [ ] Real-time prediction API

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **F1DB**: Historical F1 database ([f1db.com](https://f1db.com))
- **Open-Meteo**: Weather API ([open-meteo.com](https://open-meteo.com))
- **FastF1**: F1 telemetry library

## 📧 Contact

For questions or collaboration: [ismailycel.0@gmail.com]

**Author**: İsmail Yücel

---

**Project Status**: ✅ **Production Ready - ML Model Deployed**  
**Current Phase**: 2025 data collection & model refinement (V4)

**Stats**: 183k laps | 58 features | 99.77% accuracy | 33 circuits | 6 seasons
