"""
AeroRisk - Problem 4: SHAP Explainability
==========================================
GOAL: Explain WHY a prediction was made
TOOL: SHAP (SHapley Additive exPlanations)

Question: "Why is this flight rated HIGH RISK?"
Answer: SHAP tells you which features contributed most!

Author: Umang Kumar
Date: 2024-01-14
"""

# ============================================
# STEP 1: Import Libraries
# ============================================
import pandas as pd
import numpy as np
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# For visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("=" * 60)
print("🔬 AeroRisk - SHAP Explainability")
print("=" * 60)


# ============================================
# STEP 2: Load Trained Model and Data
# ============================================
print("\n📥 STEP 2: Loading trained model and data...")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'shap')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the XGBoost risk predictor
model_path = os.path.join(MODELS_DIR, 'xgboost_risk_predictor_v1.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"   ✅ Loaded XGBoost model from: {model_path}")
else:
    print(f"   ❌ Model not found! Run risk_predictor.py first!")
    exit(1)

# Load feature names
features_path = os.path.join(MODELS_DIR, 'xgboost_risk_features.txt')
if os.path.exists(features_path):
    with open(features_path) as f:
        feature_names = f.read().strip().split('\n')
    print(f"   ✅ Loaded {len(feature_names)} feature names")
else:
    print(f"   ⚠️  Feature names not found, using model feature names")
    feature_names = None

# Load incident data
data_path = os.path.join(DATA_DIR, 'incident_features_latest.parquet')
if os.path.exists(data_path):
    df = pd.read_parquet(data_path)
    print(f"   ✅ Loaded {len(df):,} records")
else:
    print(f"   ❌ Data not found!")
    exit(1)


# ============================================
# STEP 3: Understand SHAP
# ============================================
print("""
📚 STEP 3: Understanding SHAP Values

   ┌─────────────────────────────────────────────────────────┐
   │  WHAT ARE SHAP VALUES?                                 │
   ├─────────────────────────────────────────────────────────┤
   │  • Based on game theory (Shapley values)               │
   │  • Shows contribution of EACH feature to prediction    │
   │  • Positive SHAP = increases prediction                │
   │  • Negative SHAP = decreases prediction                │
   │                                                        │
   │  EXAMPLE:                                              │
   │  Prediction: 75 (high risk)                            │
   │  Base value: 25 (average risk)                         │
   │                                                        │
   │  Contributions:                                        │
   │  • weather_risk=80:    +20 (bad weather!)             │
   │  • crew_fatigue=90:    +15 (tired crew!)              │
   │  • maintenance_ok=1:   -10 (good maintenance)         │
   │  • aircraft_age=5:     +5  (newer but breakin period) │
   │  ─────────────────────────────────────                │
   │  Total contribution:   +30 → Final: 55                │
   └─────────────────────────────────────────────────────────┘
""")


# ============================================
# STEP 4: Prepare Data for SHAP
# ============================================
print("\n🔧 STEP 4: Preparing data for SHAP analysis...")

# Get features used by the model
if feature_names:
    available_features = [f for f in feature_names if f in df.columns]
else:
    # Get numeric columns
    available_features = df.select_dtypes(include=[np.number]).columns.tolist()
    available_features = [f for f in available_features if f not in ['injury_score', 'severity_code']]

X = df[available_features].copy()

# Fill missing values
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

print(f"   Features: {len(available_features)}")
print(f"   Samples: {len(X):,}")

# Use a sample for SHAP (SHAP can be slow on large datasets)
SAMPLE_SIZE = min(1000, len(X))
X_sample = X.sample(n=SAMPLE_SIZE, random_state=42)
print(f"   Using sample of {SAMPLE_SIZE} for SHAP analysis")


# ============================================
# STEP 5: Create SHAP Explainer
# ============================================
print("\n🔬 STEP 5: Creating SHAP explainer...")

# TreeExplainer is optimized for tree-based models (XGBoost, LightGBM)
explainer = shap.TreeExplainer(model)
print("   ✅ SHAP TreeExplainer created")

# Calculate SHAP values
print("   Calculating SHAP values (this may take a moment)...")
shap_values = explainer.shap_values(X_sample)
print(f"   ✅ SHAP values calculated! Shape: {shap_values.shape}")


# ============================================
# STEP 6: Global Feature Importance (Summary)
# ============================================
print("\n🌍 STEP 6: Global Feature Importance...")

# Calculate mean absolute SHAP values
mean_shap = pd.DataFrame({
    'feature': available_features,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\n   Top Features by SHAP Importance:")
print("   " + "-" * 50)
for i, row in mean_shap.head(10).iterrows():
    bar = "█" * int(row['mean_abs_shap'] / mean_shap['mean_abs_shap'].max() * 25)
    print(f"   {row['feature']:<25} {row['mean_abs_shap']:>6.3f} {bar}")

# Save summary plot
print("\n   📊 Generating SHAP Summary Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, feature_names=available_features, show=False)
plt.tight_layout()
summary_path = os.path.join(OUTPUT_DIR, 'shap_summary_plot.png')
plt.savefig(summary_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved to: {summary_path}")


# ============================================
# STEP 7: Individual Prediction Explanation
# ============================================
print("\n🔍 STEP 7: Explaining Individual Predictions...")

# Find high-risk and low-risk examples
predictions = model.predict(X_sample)
high_risk_idx = X_sample.iloc[predictions.argmax()].name
low_risk_idx = X_sample.iloc[predictions.argmin()].name

print("\n   --- HIGH RISK EXAMPLE ---")
high_risk_sample = X_sample.loc[[high_risk_idx]]
high_risk_pred = model.predict(high_risk_sample)[0]
high_risk_shap = explainer.shap_values(high_risk_sample)[0]

print(f"   Predicted Risk Score: {high_risk_pred:.1f}")
print(f"\n   Top Contributing Features:")

# Sort by absolute contribution
contributions = pd.DataFrame({
    'feature': available_features,
    'value': high_risk_sample.values[0],
    'shap': high_risk_shap
}).sort_values('shap', key=abs, ascending=False)

for i, row in contributions.head(5).iterrows():
    direction = "↑" if row['shap'] > 0 else "↓"
    print(f"      {direction} {row['feature']}: {row['value']:.1f} → SHAP: {row['shap']:+.2f}")


print("\n   --- LOW RISK EXAMPLE ---")
low_risk_sample = X_sample.loc[[low_risk_idx]]
low_risk_pred = model.predict(low_risk_sample)[0]
low_risk_shap = explainer.shap_values(low_risk_sample)[0]

print(f"   Predicted Risk Score: {low_risk_pred:.1f}")
print(f"\n   Top Contributing Features:")

contributions_low = pd.DataFrame({
    'feature': available_features,
    'value': low_risk_sample.values[0],
    'shap': low_risk_shap
}).sort_values('shap', key=abs, ascending=False)

for i, row in contributions_low.head(5).iterrows():
    direction = "↑" if row['shap'] > 0 else "↓"
    print(f"      {direction} {row['feature']}: {row['value']:.1f} → SHAP: {row['shap']:+.2f}")


# ============================================
# STEP 8: Generate Waterfall Plot
# ============================================
print("\n📊 STEP 8: Generating Waterfall Plots...")

# Waterfall plot for high-risk sample
plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=high_risk_shap,
        base_values=explainer.expected_value,
        data=high_risk_sample.values[0],
        feature_names=available_features
    ),
    show=False
)
plt.title("High Risk Prediction Explanation")
plt.tight_layout()
waterfall_path = os.path.join(OUTPUT_DIR, 'shap_waterfall_high_risk.png')
plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ High-risk waterfall saved to: {waterfall_path}")


# ============================================
# STEP 9: Feature Dependence Plot
# ============================================
print("\n📈 STEP 9: Generating Feature Dependence Plots...")

# Plot for top 2 features
top_features = mean_shap.head(2)['feature'].tolist()

for feat in top_features:
    if feat in X_sample.columns:
        plt.figure(figsize=(8, 5))
        feat_idx = available_features.index(feat)
        shap.dependence_plot(
            feat_idx, shap_values, X_sample,
            feature_names=available_features,
            show=False
        )
        plt.title(f"SHAP Dependence: {feat}")
        plt.tight_layout()
        dep_path = os.path.join(OUTPUT_DIR, f'shap_dependence_{feat}.png')
        plt.savefig(dep_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ {feat} dependence plot saved")


# ============================================
# STEP 10: Save SHAP Values
# ============================================
print("\n💾 STEP 10: Saving SHAP analysis results...")

# Save SHAP values
shap_df = pd.DataFrame(shap_values, columns=available_features)
shap_path = os.path.join(OUTPUT_DIR, 'shap_values.parquet')
shap_df.to_parquet(shap_path, index=False)
print(f"   ✅ SHAP values saved to: {shap_path}")

# Save feature importance
importance_path = os.path.join(OUTPUT_DIR, 'shap_feature_importance.csv')
mean_shap.to_csv(importance_path, index=False)
print(f"   ✅ Feature importance saved to: {importance_path}")


# ============================================
# STEP 11: Explainability Report
# ============================================
print("\n📋 STEP 11: Generating Explainability Report...")

report = f"""
# AeroRisk - SHAP Explainability Report

## Executive Summary
SHAP analysis reveals which features drive risk predictions.

## Top 5 Most Important Features

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
"""
for i, row in mean_shap.head(5).iterrows():
    report += f"| {mean_shap.index.get_loc(i)+1} | {row['feature']} | {row['mean_abs_shap']:.3f} |\n"

report += """
## Interpretation

- **Positive SHAP**: Feature INCREASES risk prediction
- **Negative SHAP**: Feature DECREASES risk prediction
- **Higher magnitude**: Feature has MORE impact

## Generated Visualizations

1. `shap_summary_plot.png` - Overview of all features
2. `shap_waterfall_high_risk.png` - Single prediction breakdown
3. `shap_dependence_*.png` - Feature relationships

## Usage for Safety Teams

When reviewing a high-risk prediction:
1. Look at SHAP values for that specific case
2. Identify top 3 contributing factors
3. Determine if factors are actionable
4. Generate safety recommendations based on findings
"""

report_path = os.path.join(OUTPUT_DIR, 'SHAP_REPORT.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f"   ✅ Report saved to: {report_path}")


# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("✅ SHAP EXPLAINABILITY COMPLETE!")
print("=" * 60)
print(f"""
   Tool: SHAP TreeExplainer
   Model: XGBoost Risk Predictor
   Samples analyzed: {SAMPLE_SIZE}
   
   Top Contributing Features:
""")
for i, row in mean_shap.head(5).iterrows():
    print(f"      {mean_shap.index.get_loc(i)+1}. {row['feature']}: {row['mean_abs_shap']:.3f}")

print(f"""
   
   Output Files:
   📊 {OUTPUT_DIR}/
      ├── shap_summary_plot.png
      ├── shap_waterfall_high_risk.png
      ├── shap_dependence_*.png
      ├── shap_values.parquet
      ├── shap_feature_importance.csv
      └── SHAP_REPORT.md
""")
print("=" * 60)


# ============================================
# HELPER FUNCTION: Explain Single Prediction
# ============================================
def explain_prediction(data_point, feature_names, model, explainer):
    """
    Explain a single prediction.
    
    Usage:
        explanation = explain_prediction(new_data, features, model, explainer)
    """
    shap_vals = explainer.shap_values(data_point)[0]
    pred = model.predict(data_point)[0]
    base = explainer.expected_value
    
    contributions = pd.DataFrame({
        'feature': feature_names,
        'value': data_point.values[0],
        'shap': shap_vals,
        'direction': ['↑ increases' if s > 0 else '↓ decreases' for s in shap_vals]
    }).sort_values('shap', key=abs, ascending=False)
    
    print(f"\nPrediction: {pred:.1f}")
    print(f"Base value: {base:.1f}")
    print("\nTop factors:")
    for i, row in contributions.head(5).iterrows():
        print(f"  {row['feature']}: {row['shap']:+.2f} ({row['direction']})")
    
    return contributions
