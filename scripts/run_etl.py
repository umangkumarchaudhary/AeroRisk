"""
AeroRisk - Run ETL Pipeline
Execute the ETL pipeline and generate data quality report
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from loguru import logger

from src.database.connection import engine
from src.etl.transformers import FeatureEngineer
from src.etl.validators import IncidentValidator, OperationalValidator
from src.etl.loaders import DataLoader, FeatureStore
from sqlalchemy import text


def run_etl_pipeline():
    """Run the complete ETL pipeline."""
    
    print("=" * 60)
    print("🚀 AeroRisk ETL Pipeline")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================
    # EXTRACT
    # ========================================
    print("\n📥 EXTRACT PHASE")
    print("-" * 40)
    
    # Extract incidents
    print("   Loading incidents from database...")
    with engine.connect() as conn:
        incidents_df = pd.read_sql(
            "SELECT * FROM ingestion.incidents",
            conn
        )
    print(f"   ✅ Loaded {len(incidents_df):,} incidents")
    
    # Extract operational data
    print("   Loading operational data...")
    with engine.connect() as conn:
        ops_df = pd.read_sql(
            "SELECT * FROM ingestion.operational_data",
            conn
        )
    print(f"   ✅ Loaded {len(ops_df):,} operational records")
    
    # ========================================
    # VALIDATE
    # ========================================
    print("\n✔️  VALIDATE PHASE")
    print("-" * 40)
    
    # Validate incidents
    print("   Validating incident data...")
    incident_validator = IncidentValidator()
    incident_report = incident_validator.validate(incidents_df)
    
    print(f"   Completeness: {incident_report.completeness_score:.1f}%")
    print(f"   Consistency:  {incident_report.consistency_score:.1f}%")
    print(f"   Accuracy:     {incident_report.accuracy_score:.1f}%")
    print(f"   ✅ Overall:   {incident_report.overall_score:.1f}%")
    
    # Show validation issues
    failed_checks = [v for v in incident_report.validations if not v.is_valid]
    if failed_checks:
        print(f"\n   ⚠️  {len(failed_checks)} validation issues:")
        for check in failed_checks[:5]:
            print(f"      - {check.check_name}: {check.records_failed:,} failures")
    
    # Validate operational data
    print("\n   Validating operational data...")
    ops_validator = OperationalValidator()
    ops_report = ops_validator.validate(ops_df)
    print(f"   ✅ Overall:   {ops_report.overall_score:.1f}%")
    
    # ========================================
    # TRANSFORM
    # ========================================
    print("\n🔄 TRANSFORM PHASE")
    print("-" * 40)
    
    engineer = FeatureEngineer()
    
    # Transform incidents
    print("   Engineering incident features...")
    incident_features = engineer.engineer_incident_features(incidents_df)
    new_cols = len(incident_features.columns) - len(incidents_df.columns)
    print(f"   ✅ Created {new_cols} new features")
    
    # Transform operational data
    print("   Engineering operational features...")
    ops_features = engineer.engineer_operational_features(ops_df)
    new_ops_cols = len(ops_features.columns) - len(ops_df.columns)
    print(f"   ✅ Created {new_ops_cols} new features")
    
    # Show sample features
    print("\n   📊 Sample Incident Features:")
    feature_cols = ['severity_code', 'is_serious', 'phase_risk', 
                    'injury_score', 'weather_risk']
    for col in feature_cols:
        if col in incident_features.columns:
            print(f"      {col}: mean={incident_features[col].mean():.2f}")
    
    print("\n   📊 Sample Operational Features:")
    ops_feature_cols = ['fatigue_risk', 'maintenance_risk', 
                        'schedule_risk', 'operational_risk_score']
    for col in ops_feature_cols:
        if col in ops_features.columns:
            print(f"      {col}: mean={ops_features[col].mean():.2f}")
    
    # ========================================
    # LOAD
    # ========================================
    print("\n💾 LOAD PHASE")
    print("-" * 40)
    
    store = FeatureStore()
    loader = DataLoader()
    
    # Save to feature store
    timestamp = datetime.now().strftime('%Y%m%d')
    
    print("   Saving incident features...")
    store.save_features(incident_features, f'incident_features_{timestamp}')
    store.save_features(incident_features, 'incident_features_latest')
    
    print("   Saving operational features...")
    store.save_features(ops_features, f'operational_features_{timestamp}')
    store.save_features(ops_features, 'operational_features_latest')
    
    # Log data quality
    print("   Logging data quality reports...")
    loader.log_data_quality(incident_report.to_dict())
    loader.log_data_quality(ops_report.to_dict())
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("📈 ETL PIPELINE SUMMARY")
    print("=" * 60)
    
    print(f"""
    Incidents Processed:    {len(incidents_df):,}
      - NTSB:               {len(incidents_df[incidents_df['source']=='NTSB']):,}
      - ASRS:               {len(incidents_df[incidents_df['source']=='ASRS']):,}
      - Quality Score:      {incident_report.overall_score:.1f}%
    
    Operational Records:    {len(ops_df):,}
      - Quality Score:      {ops_report.overall_score:.1f}%
    
    Features Created:
      - Incident features:  {new_cols} new columns
      - Ops features:       {new_ops_cols} new columns
    
    Output Files:
      - data/processed/incident_features_{timestamp}.parquet
      - data/processed/operational_features_{timestamp}.parquet
      - data/processed/incident_features_latest.parquet
      - data/processed/operational_features_latest.parquet
    """)
    
    print("=" * 60)
    print("✅ ETL Pipeline Complete!")
    print("=" * 60)
    
    return {
        'incidents': len(incidents_df),
        'operational': len(ops_df),
        'incident_quality': incident_report.overall_score,
        'operational_quality': ops_report.overall_score
    }


if __name__ == "__main__":
    result = run_etl_pipeline()
