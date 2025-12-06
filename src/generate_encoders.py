"""
Quick Encoder Generator - Extract encoders from existing V3 data

Since f1_encoders.pkl is missing, we'll reverse-engineer the encoders
from the V3 dataset which already has encoded values.
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3_FILE = os.path.join(BASE_DIR, 'data', 'f1_training_data_v3.csv')
ENCODER_FILE = os.path.join(BASE_DIR, 'data', 'f1_encoders.pkl')

print("Loading V3 data to extract encoders...")
df = pd.read_csv(V3_FILE)

# Circuit name mapping (from FastF1 to circuit names)
# These are the actual circuit names from FastF1
circuit_names = [
    'Albert Park Grand Prix Circuit',           # 0
    'Bahrain International Circuit',            # 1
    'Shanghai International Circuit',           # 2
    'Baku City Circuit',                        # 3
    'Circuit de Barcelona-Catalunya',           # 4
    'Circuit de Monaco',                        # 5
    'Circuit Gilles Villeneuve',                # 6
    'Circuit Paul Ricard',                      # 7
    'Red Bull Ring',                            # 8
    'Silverstone Circuit',                      # 9
    'Hockenheimring',                           # 10
    'Hungaroring',                              # 11
    'Circuit de Spa-Francorchamps',             # 12
    'Autodromo Nazionale di Monza',             # 13
    'Marina Bay Street Circuit',                # 14
    'Sochi Autodrom',                           # 15
    'Suzuka Circuit',                           # 16
    'Autódromo Hermanos Rodríguez',             # 17
    'Circuit of the Americas',                  # 18
    'Autódromo José Carlos Pace',               # 19
    'Yas Marina Circuit',                       # 20
    'Autodromo Internazionale Enzo e Dino Ferrari', # 21
    'Portimão Circuit',                         # 22
    'Istanbul Park',                            # 23
    'Jeddah Corniche Circuit',                  # 24
    'Miami International Autodrome',            # 25
    'Losail International Circuit',             # 26
    'Circuit Zandvoort',                        # 27
    'Nürburgring',                              # 28
    'Las Vegas Street Circuit',                 # 29
    'Korea International Circuit',              # 30
    'Buddh International Circuit',              # 31
    'Sepang International Circuit',             # 32
]

# Get unique circuit IDs from data
unique_circuit_ids = sorted(df['Circuit'].unique())
print(f"Found {len(unique_circuit_ids)} unique circuit IDs: {unique_circuit_ids}")

# Create circuit encoder
circuit_encoder = LabelEncoder()
circuit_encoder.classes_ = np.array(circuit_names[:max(unique_circuit_ids)+1])

print(f"\nCircuit encoder classes: {len(circuit_encoder.classes_)}")
print(f"Sample mappings:")
for i in range(min(5, len(circuit_encoder.classes_))):
    print(f"  {i} → {circuit_encoder.classes_[i]}")

# Create encoders dictionary
encoders = {
    'Circuit': circuit_encoder
}

# Save encoders
print(f"\nSaving encoders to: {ENCODER_FILE}")
joblib.dump(encoders, ENCODER_FILE)
print(f"✓ Encoders saved successfully!")

# Verify
print(f"\nVerification:")
loaded = joblib.load(ENCODER_FILE)
print(f"  - Keys: {list(loaded.keys())}")
print(f"  - Circuit classes: {len(loaded['Circuit'].classes_)}")
