# F1 Data Directory

This directory contains all data files for the project.

## Files

### f1_training_data.csv (36.99 MB)
**Final ML-ready dataset**
- 202,395 rows (lap records)
- 38 columns (features)
- Fully numeric, no missing values
- Optimized data types (float32, int32)

**Date Range**: 2019-2024 (95 races)

### f1_encoders.pkl (3.04 KB)
Label encoders for categorical features:
- Driver (36 unique drivers)
- Team (14 teams)
- Circuit (33 circuits)
- Compound (4 tire types: SOFT, MEDIUM, HARD, UNKNOWN)
- Status (47 race statuses)

**Usage**:
```python
import joblib
encoders = joblib.load('f1_encoders.pkl')
driver_name = encoders['Driver'].inverse_transform([0])[0]
```

### f1db-csv/ (Directory)
F1 historical database with 46 CSV files:
- Races, drivers, constructors, circuits
- Standings, results, lap times
- Championship data from 1950-2024

**Source**: [F1DB](https://f1db.com)

## Data Pipeline

```
Raw Data → F1DB Enrichment → Weather API → ML Preprocessing → f1_training_data.csv
```

## Download Instructions

Large files (>100MB) are not tracked in Git. Download from:
- F1DB: https://github.com/f1db/f1db/releases
- Extract to `data/f1db-csv/`

## Notes

- All CSV files use UTF-8 encoding
- Dates are in ISO format (YYYY-MM-DD)
- Missing values: 0 (after preprocessing)
