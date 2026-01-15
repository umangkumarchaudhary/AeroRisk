"""
AeroRisk - Problem 2: Severity Classification Model
====================================================
GOAL: Classify incidents into 4 severity categories
MODEL: LightGBM Classifier

Categories:
- 0 = NONE (no injuries)
- 1 = MINOR (minor injuries)
- 2 = SERIOUS (serious injuries)  
- 3 = FATAL (fatalities)

Author: Umang Kumar
Date: 2024-01-14
"""

# ============================================
# STEP 1: Import Libraries
# ============================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🎯 AeroRisk - Severity Classification (LightGBM)")
print("=" * 60)


# ============================================
# STEP 2: Load Data
# ============================================
print("\n📥 STEP 2: Loading transformed features...")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

incidents_path = os.path.join(DATA_DIR, 'incident_features_latest.parquet')

if os.path.exists(incidents_path):
    df = pd.read_parquet(incidents_path)
    print(f"   ✅ Loaded {len(df):,} records")
else:
    print(f"   ❌ File not found! Run 'python scripts/run_etl.py' first!")
    exit(1)


# ============================================
# STEP 3: Prepare Target Variable (Severity)
# ============================================
print("\n🎯 STEP 3: Preparing target variable (severity)...")

# Check what severity column exists
if 'severity' in df.columns:
    print(f"\n   Severity distribution (raw):")
    print(df['severity'].value_counts())
    
    # Encode severity to numbers
    SEVERITY_MAPPING = {
        'NONE': 0,
        'MINOR': 1,
        'SERIOUS': 2,
        'FATAL': 3
    }
    
    # Apply mapping
    df['severity_encoded'] = df['severity'].map(SEVERITY_MAPPING)
    
    # Handle any unmapped values
    df['severity_encoded'] = df['severity_encoded'].fillna(0).astype(int)
    
    TARGET_COLUMN = 'severity_encoded'
    
elif 'severity_code' in df.columns:
    TARGET_COLUMN = 'severity_code'
    print(f"   Using existing 'severity_code' column")
else:
    print("   ❌ No severity column found!")
    exit(1)

print(f"\n   Encoded severity distribution:")
print(df[TARGET_COLUMN].value_counts().sort_index())

# Count per class
class_counts = df[TARGET_COLUMN].value_counts().sort_index()
print(f"""
   ┌─────────────────────────────────────┐
   │  CLASS DISTRIBUTION                 │
   ├─────────────────────────────────────┤
   │  0 = NONE:    {class_counts.get(0, 0):>6,} records      │
   │  1 = MINOR:   {class_counts.get(1, 0):>6,} records      │
   │  2 = SERIOUS: {class_counts.get(2, 0):>6,} records      │
   │  3 = FATAL:   {class_counts.get(3, 0):>6,} records      │
   └─────────────────────────────────────┘
   
   ⚠️  Note: Classes are IMBALANCED!
   We'll use class_weight='balanced' to handle this.
""")


# ============================================
# STEP 4: Select Features
# ============================================
print("\n🔧 STEP 4: Selecting features...")

# Get all numeric columns except target
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_columns = [TARGET_COLUMN, 'severity_code', 'severity_encoded', 'injury_score']
FEATURE_COLUMNS = [col for col in numeric_columns if col not in exclude_columns]

print(f"   Using {len(FEATURE_COLUMNS)} features:")
for i, feat in enumerate(FEATURE_COLUMNS[:10]):
    print(f"      {i+1}. {feat}")
if len(FEATURE_COLUMNS) > 10:
    print(f"      ... and {len(FEATURE_COLUMNS) - 10} more")


# ============================================
# STEP 5: Prepare Training Data
# ============================================
print("\n📊 STEP 5: Preparing training data...")

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

# Fill missing values with median
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# Remove rows with missing target
mask = ~y.isnull()
X = X[mask]
y = y[mask].astype(int)

print(f"   Total samples: {len(X):,}")
print(f"   Features: {X.shape[1]}")


# ============================================
# STEP 6: Train/Test Split
# ============================================
print("\n✂️  STEP 6: Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Keep same class proportions in train/test
)

print(f"   Training set: {len(X_train):,} samples")
print(f"   Test set:     {len(X_test):,} samples")

# Show class distribution in training set
print(f"\n   Training set class distribution:")
train_dist = y_train.value_counts().sort_index()
for cls, count in train_dist.items():
    pct = count / len(y_train) * 100
    print(f"      Class {cls}: {count:,} ({pct:.1f}%)")


# ============================================
# STEP 7: Train LightGBM Classifier
# ============================================
print("\n🏋️ STEP 7: Training LightGBM classifier...")

# Create LightGBM Classifier with balanced class weights
model = lgb.LGBMClassifier(
    n_estimators=100,           # Number of trees
    max_depth=8,                # Maximum tree depth
    learning_rate=0.1,          # Step size
    num_leaves=31,              # LightGBM specific (2^depth - 1)
    class_weight='balanced',    # HANDLE IMBALANCED CLASSES!
    random_state=42,
    n_jobs=-1,                  # Use all CPU cores
    verbose=-1                  # Silence output
)

print("   Training with class_weight='balanced'...")
model.fit(X_train, y_train)
print("   ✅ Training complete!")


# ============================================
# STEP 8: Evaluate Model
# ============================================
print("\n📈 STEP 8: Evaluating model performance...")

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"""
   ┌─────────────────────────────────────┐
   │  CLASSIFICATION METRICS             │
   ├─────────────────────────────────────┤
   │  Accuracy:         {accuracy:.1%}          │
   │  F1 Score (macro): {f1_macro:.4f}           │
   │  F1 Score (weighted): {f1_weighted:.4f}        │
   └─────────────────────────────────────┘
   
   Note: For imbalanced classes, F1 is more 
   important than accuracy!
""")


# ============================================
# STEP 9: Confusion Matrix
# ============================================
print("\n📊 STEP 9: Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)
class_names = ['NONE', 'MINOR', 'SERIOUS', 'FATAL']

print("""
   Rows = Actual, Columns = Predicted
   
                     Predicted
                NONE  MINOR SERIOUS FATAL
   Actual NONE   [{:>4}]  {:>4}   {:>4}   {:>4}
         MINOR   {:>4}  [{:>4}]  {:>4}   {:>4}
        SERIOUS  {:>4}   {:>4}  [{:>4}]  {:>4}
         FATAL   {:>4}   {:>4}   {:>4}  [{:>4}]
         
   Numbers in [brackets] = correctly classified
""".format(
    cm[0,0], cm[0,1], cm[0,2], cm[0,3],
    cm[1,0], cm[1,1], cm[1,2], cm[1,3],
    cm[2,0], cm[2,1], cm[2,2], cm[2,3],
    cm[3,0], cm[3,1], cm[3,2], cm[3,3]
))


# ============================================
# STEP 10: Classification Report
# ============================================
print("\n📋 STEP 10: Detailed Classification Report...")

report = classification_report(y_test, y_pred, target_names=class_names)
print(report)


# ============================================
# STEP 11: Feature Importance
# ============================================
print("\n🏆 STEP 11: Top 10 Most Important Features...")

importance = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   " + "-" * 45)
for i, row in importance.head(10).iterrows():
    bar = "█" * int(row['importance'] / importance['importance'].max() * 20)
    print(f"   {row['feature'][:25]:<25} {row['importance']:>6.0f} {bar}")


# ============================================
# STEP 12: Save Model
# ============================================
print("\n💾 STEP 12: Saving model...")

model_path = os.path.join(MODELS_DIR, 'lightgbm_severity_classifier_v1.pkl')
joblib.dump(model, model_path)
print(f"   ✅ Model saved to: {model_path}")

# Save class mapping
mapping_path = os.path.join(MODELS_DIR, 'severity_class_mapping.txt')
with open(mapping_path, 'w') as f:
    for name, code in SEVERITY_MAPPING.items():
        f.write(f"{code}: {name}\n")
print(f"   ✅ Class mapping saved to: {mapping_path}")


# ============================================
# STEP 13: Example Predictions
# ============================================
print("\n🔮 STEP 13: Example Predictions...")

print("\n   Sample predictions vs actual:")
print("   " + "-" * 40)
sample_indices = X_test.sample(5).index
for idx in sample_indices:
    actual_code = y_test.loc[idx]
    predicted_code = model.predict(X_test.loc[[idx]])[0]
    
    actual_name = class_names[actual_code]
    predicted_name = class_names[predicted_code]
    
    match = "✓" if actual_code == predicted_code else "✗"
    print(f"   Actual: {actual_name:<8} Predicted: {predicted_name:<8} {match}")


# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("✅ SEVERITY CLASSIFICATION MODEL COMPLETE!")
print("=" * 60)
print(f"""
   Model: LightGBM Classifier
   Classes: NONE, MINOR, SERIOUS, FATAL
   Features: {len(FEATURE_COLUMNS)}
   
   Performance:
   - Accuracy:     {accuracy:.1%}
   - F1 (macro):   {f1_macro:.4f}
   - F1 (weighted): {f1_weighted:.4f}
   
   Class weights: BALANCED (handles imbalance)
   
   Saved to: {model_path}
""")
print("=" * 60)


# ============================================
# HOW TO USE THIS MODEL
# ============================================
"""
To load and use this model:

import joblib
import pandas as pd

# Load model
model = joblib.load('models/lightgbm_severity_classifier_v1.pkl')

# Class names
classes = ['NONE', 'MINOR', 'SERIOUS', 'FATAL']

# Prepare new data
new_data = pd.DataFrame({
    'incident_year': [2024],
    'weather_risk_score': [65],
    # ... other features
})

# Predict
prediction = model.predict(new_data)[0]
probability = model.predict_proba(new_data)[0]

print(f"Predicted severity: {classes[prediction]}")
print(f"Confidence: {probability[prediction]:.1%}")
"""
