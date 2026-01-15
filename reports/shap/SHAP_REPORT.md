
# AeroRisk - SHAP Explainability Report

## Executive Summary
SHAP analysis reveals which features drive risk predictions.

## Top 5 Most Important Features

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | total_injuries | 3.070 |
| 2 | severity_code | 1.638 |
| 3 | injury_rate | 0.847 |
| 4 | is_us | 0.104 |
| 5 | is_weekend | 0.037 |

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
