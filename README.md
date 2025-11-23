# F1 Race Analysis & Machine Learning Pipeline

A comprehensive data science project analyzing Formula 1 race data (2019-2024) with machine learning preprocessing pipeline.

## 📊 Project Overview

This project collects, enriches, and prepares Formula 1 racing data for machine learning models. It integrates multiple data sources including:
- FastF1 telemetry data
- F1DB historical database
- Open-Meteo weather API

**Final Dataset**: 202,395 race laps across 95 races with 38 features

## 🚀 Features

- **Multi-source Data Integration**: Combines F1DB database, weather data, and telemetry
- **Comprehensive Enrichment**: Driver demographics, championship standings, circuit data, weather conditions
- **ML-Ready Output**: Fully numeric dataset with optimized data types
- **Automated Pipeline**: End-to-end data processing with error handling
- **Performance Optimized**: SQLite caching, memory optimization (float32/int32)

## 📁 Project Structure

```
f1-race-analysis/
├── data/
│   ├── f1_training_data.csv          # Final ML-ready dataset (37 MB)
│   ├── f1_encoders.pkl                # Label encoders for categorical variables
│   └── f1db-csv/                      # F1 historical database (46 CSV files)
│
├── src/
│   ├── f1_database_enrichment.py      # F1DB data enrichment pipeline
│   ├── f1_weather_enrichment.py       # Weather API integration
│   ├── f1_preprocessing.py            # ML preprocessing pipeline
│   └── f1_data_analysis.py            # Data analysis & visualization
│
├── output/
│   ├── analysis_output.txt            # Dataset statistics
│   └── correlation_results.txt        # Feature correlation analysis
│
├── cache/
│   └── weather_cache.sqlite           # API response cache
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
git clone https://github.com/ismailyucel/f1-race-analysis.git
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
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 🎯 Usage

### 1. Data Enrichment (F1DB Integration)

```bash
python src/f1_ultimate_enrichment.py
```

**Adds**: Driver age, championship standings, circuit coordinates

### 2. Weather Data Collection

```bash
python src/f1_weather_enrichment.py
```

**Adds**: Temperature, precipitation, pressure, wind speed/direction

### 3. ML Preprocessing

```bash
python src/f1_ml_preprocessing.py
```

**Output**: `f1_training_data.csv` - fully numeric, no missing values

### 4. Data Analysis

```bash
python src/veri_analizi_ve_gorsellestirme.py
```

**Generates**: Correlation matrices, distribution plots, statistical summaries

## 📊 Dataset Details

### Final Dataset Specifications
- **Rows**: 202,395 lap records
- **Columns**: 38 features
- **Size**: 36.99 MB
- **Format**: CSV (float32/int32 optimized)
- **Missing Values**: 0
- **Date Range**: 2019-2024 (95 races)

### Feature Categories

**Telemetry (9)**:
- LapTime, Sector1Time, Sector2Time, Sector3Time
- SpeedST, Position, TrackTemp, AirTemp, Humidity

**Race Context (8)**:
- Year, Round, Circuit, GridPosition, Status, Points, GapToLeader, Rainfall

**Driver/Team (4)**:
- Driver (encoded), Team (encoded), DriverAge, DriverStandingsPoints/Position

**Weather (5)**:
- WeatherTemp, WeatherPrecipitation, WeatherPressure, WeatherWindSpeed, WeatherWindDirection

**Tire Strategy (4)**:
- Compound (encoded), TyreLife, FreshTyre, Stint

**Circuit Data (4)**:
- CircuitLat, CircuitLng, ConstructorStandingsPoints/Position

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

### Performance Optimizations
- **Memory**: 30.88 MB RAM usage (vs 68 MB raw)
- **API Caching**: SQLite cache prevents duplicate requests
- **Data Types**: float64→float32, int64→int32 conversions

### Error Handling
- Try-except blocks for API failures
- Rate limiting (0.2s delay between requests)
- Graceful degradation for missing data

### Code Quality
- Type hints and docstrings
- Modular pipeline design
- Progress tracking with detailed logging

## 📝 Encoders

Categorical features are label-encoded and saved in `f1_encoders.pkl`:

```python
import joblib
encoders = joblib.load('data/f1_encoders.pkl')

# Decode driver ID
driver_name = encoders['Driver'].inverse_transform([0])[0]

# Available encoders: Driver, Team, Circuit, Compound, Status
```

## 🔬 Future Work

- [ ] Feature engineering (rolling averages, pace deltas)
- [ ] Train/test split (80/20)
- [ ] Model training (XGBoost, LSTM)
- [ ] Hyperparameter tuning
- [ ] Position prediction accuracy metrics
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

**Project Status**: Data collection and preprocessing complete ✅  
**Next Phase**: Model training and evaluation
