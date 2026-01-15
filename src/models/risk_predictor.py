"""
AeroRisk - Problem 1: Risk Prediction Model
============================================
GOAL: Predict flight/operation risk score (0-100)
MODEL: XGBoost Regressor

Author: Your Name
Date: 2024-01-14
"""

# ============================================
# STEP 1: Import Libraries
# ============================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 AeroRisk - Risk Prediction Model (XGBoost)")
print("=" * 60)


# ============================================
# STEP 2: Load Data from Parquet Files
# ============================================
print("\n📥 STEP 2: Loading transformed features...")

# Path to processed data (created by run_etl.py)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

# Create models directory if doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load the incident features (our main dataset)
incidents_path = os.path.join(DATA_DIR, 'incident_features_latest.parquet')

if os.path.exists(incidents_path):
    df = pd.read_parquet(incidents_path)
    print(f"   ✅ Loaded {len(df):,} records with {len(df.columns)} columns")
else:
    print(f"   ❌ File not found: {incidents_path}")
    print("   Run 'python scripts/run_etl.py' first!")
    exit(1)


# ============================================
# STEP 3: Explore the Data
# ============================================
print("\n📊 STEP 3: Data Exploration...")
print(f"\n   Available columns:")
for col in sorted(df.columns):
    dtype = df[col].dtype
    nulls = df[col].isnull().sum()
    print(f"      - {col}: {dtype} ({nulls} nulls)")


# ============================================
# STEP 4: Select Features and Target
# ============================================
print("\n🎯 STEP 4: Selecting features and target...")

# TARGET: What we want to predict
# We'll use 'injury_score' or 'severity_code' as our risk proxy
# injury_score = fatal*5 + serious*3 + minor*1 (created in transformers.py)

TARGET_COLUMN = 'injury_score'  # Change this if you want different target

# Check if target exists
if TARGET_COLUMN not in df.columns:
    print(f"   ⚠️  '{TARGET_COLUMN}' not found. Available numeric columns:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        print(f"      - {col}")
    # Use severity_code as fallback
    TARGET_COLUMN = 'severity_code' if 'severity_code' in df.columns else numeric_cols[0]
    print(f"   Using '{TARGET_COLUMN}' as target instead")

# FEATURES: What we use to predict
# Select numeric columns that make sense for risk prediction
FEATURE_COLUMNS = [
    # Temporal features
    'incident_year', 'incident_month', 'incident_day_of_week',
    'is_weekend',
    
    # Severity encoding
    'severity_code',
    
    # Phase of flight risk
    'phase_risk_factor',
    
    # Injury data
    'total_injuries', 'injury_rate',
    
    # Location features
    'has_coordinates', 'is_us',
    
    # Weather features (from weather enrichment)
    'weather_risk_score',
]

# Filter to only columns that exist in our data
available_features = [col for col in FEATURE_COLUMNS if col in df.columns]

# If we don't have enough features, add any numeric columns
if len(available_features) < 5:
    print("   ⚠️  Not enough predefined features, adding all numeric columns...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available_features = [col for col in numeric_cols if col != TARGET_COLUMN]

print(f"\n   Target: {TARGET_COLUMN}")
print(f"   Features ({len(available_features)}):")
for feat in available_features:
    print(f"      - {feat}")


# ============================================
# STEP 5: Prepare Training Data
# ============================================
print("\n🔧 STEP 5: Preparing training data...")

# Create feature matrix (X) and target vector (y)
X = df[available_features].copy()
y = df[TARGET_COLUMN].copy()

# Handle missing values
print(f"\n   Before cleaning: {len(X):,} rows")
print(f"   Missing values per column:")
for col in X.columns:
    missing = X[col].isnull().sum()
    if missing > 0:
        print(f"      - {col}: {missing} missing")

# Fill missing numeric values with median
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# Remove rows where target is missing
mask = ~y.isnull()
X = X[mask]
y = y[mask]

print(f"   After cleaning: {len(X):,} rows")


# ============================================
# STEP 6: Split into Train/Test Sets
# ============================================
print("\n✂️  STEP 6: Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # For reproducibility
)

print(f"   Training set: {len(X_train):,} samples")
print(f"   Test set:     {len(X_test):,} samples")


# ============================================
# STEP 7: Train XGBoost Model
# ============================================
print("\n🏋️ STEP 7: Training XGBoost model...")

# Create XGBoost Regressor with good default parameters
model = xgb.XGBRegressor(
    n_estimators=100,       # Number of trees
    max_depth=6,            # Maximum tree depth
    learning_rate=0.1,      # Step size shrinkage
    subsample=0.8,          # % of samples per tree
    colsample_bytree=0.8,   # % of features per tree
    random_state=42,
    n_jobs=-1,              # Use all CPU cores
    verbosity=0             # Silence warnings
)

# Train the model
print("   Training...")
model.fit(X_train, y_train)
print("   ✅ Training complete!")


# ============================================
# STEP 8: Evaluate Model Performance
# ============================================
print("\n📈 STEP 8: Evaluating model performance...")

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"""
   ┌─────────────────────────────────────┐
   │  MODEL PERFORMANCE METRICS          │
   ├─────────────────────────────────────┤
   │  RMSE (Root Mean Squared Error):    │
   │      {rmse:.4f}                      
   │  (Lower is better, ideal = 0)       │
   │                                     │
   │  MAE (Mean Absolute Error):         │
   │      {mae:.4f}                       
   │  (Average prediction error)         │
   │                                     │
   │  R² Score (Coefficient of Det.):    │
   │      {r2:.4f}                        
   │  (1.0 = perfect, 0 = baseline)      │
   └─────────────────────────────────────┘
""")

# Interpret results
if r2 > 0.8:
    print("   🎉 Excellent model! R² > 0.8")
elif r2 > 0.6:
    print("   👍 Good model. R² > 0.6")
elif r2 > 0.4:
    print("   ⚠️  Moderate model. Consider more features.")
else:
    print("   ❌ Poor model. Needs improvement.")


# ============================================
# STEP 9: Feature Importance
# ============================================
print("\n🏆 STEP 9: Feature Importance (What matters most?)...")

# Get feature importance scores
importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 Most Important Features:")
print("   " + "-" * 40)
for i, row in importance.head(10).iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"   {row['feature'][:25]:<25} {row['importance']:.3f} {bar}")


# ============================================
# STEP 10: Save Model
# ============================================
print("\n💾 STEP 10: Saving model...")

model_path = os.path.join(MODELS_DIR, 'xgboost_risk_predictor_v1.pkl')
joblib.dump(model, model_path)
print(f"   ✅ Model saved to: {model_path}")

# Also save feature names for later use
feature_path = os.path.join(MODELS_DIR, 'xgboost_risk_features.txt')
with open(feature_path, 'w') as f:
    f.write('\n'.join(available_features))
print(f"   ✅ Features saved to: {feature_path}")


# ============================================
# STEP 11: Test Prediction (Demo)
# ============================================
print("\n🔮 STEP 11: Example Predictions...")

# Show some example predictions
print("\n   Sample predictions vs actual values:")
print("   " + "-" * 50)
sample_indices = X_test.sample(5).index
for idx in sample_indices:
    actual = y_test.loc[idx]
    predicted = model.predict(X_test.loc[[idx]])[0]
    diff = predicted - actual
    print(f"   Actual: {actual:6.1f}  Predicted: {predicted:6.1f}  Diff: {diff:+.1f}")


# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("✅ RISK PREDICTION MODEL COMPLETE!")
print("=" * 60)
print(f"""
   Model: XGBoost Regressor
   Target: {TARGET_COLUMN}
   Features: {len(available_features)}
   Training samples: {len(X_train):,}
   Test samples: {len(X_test):,}
   
   Performance:
   - RMSE: {rmse:.4f}
   - R²:   {r2:.4f}
   
   Saved to: {model_path}
""")
print("=" * 60)


# ============================================
# HOW TO USE THIS MODEL LATER
# ============================================
"""
To load and use this model:

import joblib
import pandas as pd

# Load the model
model = joblib.load('models/xgboost_risk_predictor_v1.pkl')

# Prepare new data (must have same features!)
new_data = pd.DataFrame({
    'incident_year': [2024],
    'incident_month': [6],
    'weather_risk_score': [75],
    # ... other features
})

# Make prediction
risk_score = model.predict(new_data)[0]
print(f"Predicted risk score: {risk_score:.1f}")
"""
