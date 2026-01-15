"""
AeroRisk - Problem 3: Anomaly Detection Model
==============================================
GOAL: Find unusual/suspicious operational patterns
MODEL: Isolation Forest (Unsupervised)

Use Case: "This flight's risk factors look abnormal!"
Output: is_anomaly (True/False), anomaly_score (-1 to 0)

Author: Umang Kumar
Date: 2024-01-14
"""

# ============================================
# STEP 1: Import Libraries
# ============================================
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🔍 AeroRisk - Anomaly Detection (Isolation Forest)")
print("=" * 60)


# ============================================
# STEP 2: Load Operational Data
# ============================================
print("\n📥 STEP 2: Loading operational data...")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Load operational data (50,000 synthetic records)
ops_path = os.path.join(DATA_DIR, 'operational_features_latest.parquet')

if os.path.exists(ops_path):
    df = pd.read_parquet(ops_path)
    print(f"   ✅ Loaded {len(df):,} operational records")
else:
    print(f"   ⚠️  Operational data not found at: {ops_path}")
    print("   Trying incident data instead...")
    
    # Fallback to incident data
    inc_path = os.path.join(DATA_DIR, 'incident_features_latest.parquet')
    if os.path.exists(inc_path):
        df = pd.read_parquet(inc_path)
        print(f"   ✅ Loaded {len(df):,} incident records")
    else:
        print("   ❌ No data found! Run 'python scripts/run_etl.py' first!")
        exit(1)


# ============================================
# STEP 3: Understand Anomaly Detection
# ============================================
print("""
📚 STEP 3: Understanding Anomaly Detection
   
   ┌─────────────────────────────────────────────────────┐
   │  WHAT IS ISOLATION FOREST?                         │
   ├─────────────────────────────────────────────────────┤
   │  • Unsupervised learning (no labels needed!)       │
   │  • Finds points that are DIFFERENT from majority   │
   │  • Works by "isolating" outliers with random cuts  │
   │                                                    │
   │  HOW IT WORKS:                                     │
   │  1. Randomly select a feature                      │
   │  2. Randomly select a split value                  │
   │  3. Repeat until point is isolated                 │
   │                                                    │
   │  KEY INSIGHT:                                      │
   │  Anomalies need FEWER splits to isolate!           │
   │  (Because they're far from normal points)          │
   └─────────────────────────────────────────────────────┘
""")


# ============================================
# STEP 4: Select Features for Anomaly Detection
# ============================================
print("\n🔧 STEP 4: Selecting features for anomaly detection...")

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Define features we want to check for anomalies
# These are operational risk factors
ANOMALY_FEATURES = [
    # Risk indicators (if available)
    'operational_risk_score',
    'fatigue_risk',
    'maintenance_risk',
    'schedule_risk',
    
    # Crew factors
    'crew_duty_hours',
    'crew_rest_hours',
    
    # Operations
    'schedule_deviation_mins',
    'turnaround_time_mins',
    
    # From incident data (if using incidents)
    'total_injuries',
    'injury_score',
    'weather_risk_score',
    'phase_risk_factor',
]

# Filter to available columns
available_features = [col for col in ANOMALY_FEATURES if col in df.columns]

# If not enough features, use all numeric
if len(available_features) < 3:
    print("   ⚠️  Not enough predefined features, using all numeric columns...")
    exclude = ['id', 'year', 'month', 'day']
    available_features = [col for col in numeric_cols if not any(x in col.lower() for x in exclude)]

print(f"\n   Using {len(available_features)} features for anomaly detection:")
for feat in available_features[:10]:
    print(f"      • {feat}")
if len(available_features) > 10:
    print(f"      ... and {len(available_features) - 10} more")


# ============================================
# STEP 5: Prepare Data
# ============================================
print("\n📊 STEP 5: Preparing data...")

X = df[available_features].copy()

# Fill missing values with median
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# Standardize features (important for anomaly detection!)
print("   Scaling features (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   Total samples: {len(X):,}")
print(f"   Features: {X.shape[1]}")


# ============================================
# STEP 6: Train Isolation Forest
# ============================================
print("\n🏋️ STEP 6: Training Isolation Forest...")

# contamination = expected % of anomalies
# We set 5% (0.05) to find the weirdest 5%
CONTAMINATION = 0.05

model = IsolationForest(
    n_estimators=100,           # Number of trees
    max_samples='auto',         # Samples per tree
    contamination=CONTAMINATION, # Expected anomaly rate
    random_state=42,
    n_jobs=-1,                  # Use all CPU cores
    verbose=0
)

print(f"   Contamination rate: {CONTAMINATION:.0%} (expecting {int(len(X) * CONTAMINATION):,} anomalies)")
print("   Training...")
model.fit(X_scaled)
print("   ✅ Training complete!")


# ============================================
# STEP 7: Predict Anomalies
# ============================================
print("\n🔮 STEP 7: Detecting anomalies...")

# Predict: 1 = normal, -1 = anomaly
predictions = model.predict(X_scaled)

# Get anomaly scores (more negative = more abnormal)
anomaly_scores = model.decision_function(X_scaled)

# Add results to dataframe
df['is_anomaly'] = predictions == -1  # True if anomaly
df['anomaly_score'] = anomaly_scores

# Count results
n_anomalies = (predictions == -1).sum()
n_normal = (predictions == 1).sum()

print(f"""
   ┌─────────────────────────────────────┐
   │  ANOMALY DETECTION RESULTS          │
   ├─────────────────────────────────────┤
   │  Normal records:   {n_normal:>6,}          │
   │  Anomalies found:  {n_anomalies:>6,} ({n_anomalies/len(df)*100:.1f}%)    │
   └─────────────────────────────────────┘
""")


# ============================================
# STEP 8: Analyze Anomalies
# ============================================
print("\n📊 STEP 8: Analyzing anomalies...")

# Get anomaly and normal subsets
anomalies_df = df[df['is_anomaly'] == True]
normal_df = df[df['is_anomaly'] == False]

# Compare statistics
print("\n   Feature comparison (Anomalies vs Normal):")
print("   " + "-" * 55)
print(f"   {'Feature':<25} {'Anomaly Avg':>12} {'Normal Avg':>12} {'Diff':>10}")
print("   " + "-" * 55)

for feat in available_features[:8]:
    if feat in anomalies_df.columns:
        anomaly_mean = anomalies_df[feat].mean()
        normal_mean = normal_df[feat].mean()
        diff = anomaly_mean - normal_mean
        diff_pct = (diff / normal_mean * 100) if normal_mean != 0 else 0
        
        indicator = "🔴" if abs(diff_pct) > 50 else "🟡" if abs(diff_pct) > 20 else "🟢"
        print(f"   {indicator} {feat:<23} {anomaly_mean:>12.1f} {normal_mean:>12.1f} {diff_pct:>+9.1f}%")


# ============================================
# STEP 9: Top 10 Most Anomalous Records
# ============================================
print("\n🚨 STEP 9: Top 10 Most Anomalous Records...")

# Sort by anomaly score (most negative = most anomalous)
top_anomalies = df.nsmallest(10, 'anomaly_score')

print("\n   Most abnormal operations:")
print("   " + "-" * 60)
for i, (idx, row) in enumerate(top_anomalies.iterrows()):
    score = row['anomaly_score']
    # Show key feature values if available
    detail = ""
    if 'operational_risk_score' in row:
        detail = f" | Risk: {row['operational_risk_score']:.0f}"
    elif 'injury_score' in row:
        detail = f" | Injury Score: {row['injury_score']:.0f}"
    
    severity = "🔴🔴🔴" if i < 3 else "🔴🔴" if i < 6 else "🔴"
    print(f"   {i+1:>2}. {severity} Score: {score:.3f}{detail}")


# ============================================
# STEP 10: Save Model and Results
# ============================================
print("\n💾 STEP 10: Saving model and results...")

# Save model
model_path = os.path.join(MODELS_DIR, 'isolation_forest_anomaly_v1.pkl')
joblib.dump(model, model_path)
print(f"   ✅ Model saved to: {model_path}")

# Save scaler (needed for new predictions)
scaler_path = os.path.join(MODELS_DIR, 'anomaly_scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"   ✅ Scaler saved to: {scaler_path}")

# Save feature names
features_path = os.path.join(MODELS_DIR, 'anomaly_features.txt')
with open(features_path, 'w') as f:
    f.write('\n'.join(available_features))
print(f"   ✅ Features saved to: {features_path}")

# Save anomaly results
results_path = os.path.join(DATA_DIR, 'anomaly_detection_results.parquet')
df[['is_anomaly', 'anomaly_score']].to_parquet(results_path, index=False)
print(f"   ✅ Results saved to: {results_path}")


# ============================================
# STEP 11: Anomaly Score Distribution
# ============================================
print("\n📊 STEP 11: Anomaly Score Distribution...")

# Show distribution of anomaly scores
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("\n   Anomaly Score Percentiles:")
print("   (More negative = more abnormal)")
print("   " + "-" * 35)
for p in percentiles:
    val = np.percentile(anomaly_scores, p)
    marker = "← Threshold" if p == 5 else ""
    print(f"   {p:>3}th percentile: {val:.4f} {marker}")


# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("✅ ANOMALY DETECTION MODEL COMPLETE!")
print("=" * 60)
print(f"""
   Model: Isolation Forest
   Contamination: {CONTAMINATION:.0%}
   Features: {len(available_features)}
   Total records: {len(df):,}
   
   Results:
   - Normal:    {n_normal:,} records ({n_normal/len(df)*100:.1f}%)
   - Anomalies: {n_anomalies:,} records ({n_anomalies/len(df)*100:.1f}%)
   
   Saved to: {model_path}
   
   Key Finding:
   Anomalous records have significantly DIFFERENT
   feature values compared to normal operations!
""")
print("=" * 60)


# ============================================
# HOW TO USE THIS MODEL
# ============================================
"""
To detect anomalies in new data:

import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model = joblib.load('models/isolation_forest_anomaly_v1.pkl')
scaler = joblib.load('models/anomaly_scaler.pkl')

# Load feature names
with open('models/anomaly_features.txt') as f:
    features = f.read().strip().split('\n')

# Prepare new data
new_data = pd.DataFrame({
    'operational_risk_score': [85],  # Unusually high!
    'fatigue_risk': [90],            # Very high fatigue
    # ... other features
})

# Scale the data
X_new_scaled = scaler.transform(new_data[features])

# Predict
prediction = model.predict(X_new_scaled)[0]
score = model.decision_function(X_new_scaled)[0]

if prediction == -1:
    print(f"⚠️ ANOMALY DETECTED! Score: {score:.3f}")
else:
    print(f"✓ Normal operation. Score: {score:.3f}")
"""
