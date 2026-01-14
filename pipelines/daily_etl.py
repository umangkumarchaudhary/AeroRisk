"""
AeroRisk - Prefect Daily Ingestion Flow
Orchestrated ETL pipeline for data processing
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
import pandas as pd

from src.database.connection import engine
from src.etl.transformers import FeatureEngineer
from src.etl.validators import IncidentValidator, OperationalValidator
from src.etl.loaders import DataLoader, FeatureStore
from sqlalchemy import text


@task(
    name="Extract Incidents",
    retries=2,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def extract_incidents(days_back: int = 365) -> pd.DataFrame:
    """Extract incident data from database."""
    logger = get_run_logger()
    logger.info(f"Extracting incidents from last {days_back} days...")
    
    query = text("""
        SELECT * FROM ingestion.incidents
        WHERE incident_date >= CURRENT_DATE - :days
        ORDER BY incident_date DESC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={'days': days_back})
    
    logger.info(f"Extracted {len(df):,} incidents")
    return df


@task(
    name="Extract Operational Data",
    retries=2,
    retry_delay_seconds=60
)
def extract_operational_data(days_back: int = 90) -> pd.DataFrame:
    """Extract operational data from database."""
    logger = get_run_logger()
    logger.info(f"Extracting operational data from last {days_back} days...")
    
    query = text("""
        SELECT * FROM ingestion.operational_data
        WHERE date >= CURRENT_DATE - :days
        ORDER BY date DESC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={'days': days_back})
    
    logger.info(f"Extracted {len(df):,} operational records")
    return df


@task(name="Validate Incidents")
def validate_incidents(df: pd.DataFrame) -> dict:
    """Validate incident data quality."""
    logger = get_run_logger()
    logger.info("Validating incident data...")
    
    validator = IncidentValidator()
    report = validator.validate(df)
    
    logger.info(f"Validation complete - Score: {report.overall_score:.1f}%")
    
    # Log failed checks
    for v in report.validations:
        if not v.is_valid:
            logger.warning(f"Check '{v.check_name}' failed: {v.records_failed:,} records")
    
    return report.to_dict()


@task(name="Validate Operational Data")
def validate_operational(df: pd.DataFrame) -> dict:
    """Validate operational data quality."""
    logger = get_run_logger()
    logger.info("Validating operational data...")
    
    validator = OperationalValidator()
    report = validator.validate(df)
    
    logger.info(f"Validation complete - Score: {report.overall_score:.1f}%")
    return report.to_dict()


@task(name="Transform Incidents")
def transform_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """Transform and engineer incident features."""
    logger = get_run_logger()
    logger.info("Transforming incident data...")
    
    engineer = FeatureEngineer()
    features_df = engineer.engineer_incident_features(df)
    
    logger.info(f"Created {len(features_df.columns)} features for {len(features_df):,} records")
    return features_df


@task(name="Transform Operational Data")
def transform_operational(df: pd.DataFrame) -> pd.DataFrame:
    """Transform and engineer operational features."""
    logger = get_run_logger()
    logger.info("Transforming operational data...")
    
    engineer = FeatureEngineer()
    features_df = engineer.engineer_operational_features(df)
    
    logger.info(f"Created {len(features_df.columns)} features for {len(features_df):,} records")
    return features_df


@task(name="Save Features")
def save_features(
    incident_features: pd.DataFrame,
    operational_features: pd.DataFrame
) -> dict:
    """Save engineered features to feature store."""
    logger = get_run_logger()
    logger.info("Saving features to feature store...")
    
    store = FeatureStore()
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    
    incident_count = store.save_features(
        incident_features,
        f'incident_features_{timestamp}'
    )
    
    ops_count = store.save_features(
        operational_features,
        f'operational_features_{timestamp}'
    )
    
    # Also save latest version
    store.save_features(incident_features, 'incident_features_latest')
    store.save_features(operational_features, 'operational_features_latest')
    
    logger.info(f"Saved {incident_count:,} incident and {ops_count:,} operational features")
    
    return {
        'incident_features': incident_count,
        'operational_features': ops_count,
        'timestamp': timestamp
    }


@task(name="Log Data Quality")
def log_quality_reports(
    incident_report: dict,
    operational_report: dict
) -> None:
    """Log data quality reports to database."""
    logger = get_run_logger()
    logger.info("Logging data quality reports...")
    
    loader = DataLoader()
    loader.log_data_quality(incident_report)
    loader.log_data_quality(operational_report)
    
    logger.info("Quality reports logged successfully")


@task(name="Generate Summary Report")
def generate_summary(
    incident_features: pd.DataFrame,
    operational_features: pd.DataFrame,
    incident_report: dict,
    feature_info: dict
) -> dict:
    """Generate ETL pipeline summary report."""
    logger = get_run_logger()
    
    summary = {
        'run_timestamp': datetime.now().isoformat(),
        'incidents': {
            'total_records': len(incident_features),
            'features_created': len(incident_features.columns),
            'quality_score': incident_report['overall_score'],
            'by_source': incident_features['source'].value_counts().to_dict() if 'source' in incident_features.columns else {},
            'by_severity': incident_features['severity'].value_counts().to_dict() if 'severity' in incident_features.columns else {}
        },
        'operational': {
            'total_records': len(operational_features),
            'features_created': len(operational_features.columns)
        },
        'feature_store': feature_info,
        'status': 'SUCCESS'
    }
    
    logger.info("=" * 50)
    logger.info("ETL Pipeline Summary")
    logger.info("=" * 50)
    logger.info(f"Incidents processed: {summary['incidents']['total_records']:,}")
    logger.info(f"Operational records: {summary['operational']['total_records']:,}")
    logger.info(f"Quality score: {summary['incidents']['quality_score']:.1f}%")
    logger.info("=" * 50)
    
    return summary


@flow(
    name="AeroRisk Daily ETL",
    description="Daily ETL pipeline for incident and operational data",
    version="1.0.0"
)
def daily_etl_flow(
    incident_days_back: int = 365,
    operational_days_back: int = 90
) -> dict:
    """
    Main ETL flow for AeroRisk data processing.
    
    Args:
        incident_days_back: Number of days of incident data to process
        operational_days_back: Number of days of operational data to process
        
    Returns:
        Pipeline summary report
    """
    logger = get_run_logger()
    logger.info("🚀 Starting AeroRisk Daily ETL Pipeline")
    
    # Extract
    incidents_df = extract_incidents(incident_days_back)
    operational_df = extract_operational_data(operational_days_back)
    
    # Validate
    incident_report = validate_incidents(incidents_df)
    operational_report = validate_operational(operational_df)
    
    # Transform
    incident_features = transform_incidents(incidents_df)
    operational_features = transform_operational(operational_df)
    
    # Load
    feature_info = save_features(incident_features, operational_features)
    log_quality_reports(incident_report, operational_report)
    
    # Generate summary
    summary = generate_summary(
        incident_features,
        operational_features,
        incident_report,
        feature_info
    )
    
    logger.info("✅ ETL Pipeline completed successfully!")
    return summary


@flow(
    name="AeroRisk Full Refresh",
    description="Full data refresh - processes all historical data"
)
def full_refresh_flow() -> dict:
    """Full refresh flow - process all available data."""
    return daily_etl_flow(
        incident_days_back=36500,  # ~100 years
        operational_days_back=365
    )


if __name__ == "__main__":
    # Run the daily flow
    result = daily_etl_flow()
    print("\n📊 Pipeline Result:")
    print(f"   Status: {result['status']}")
    print(f"   Incidents: {result['incidents']['total_records']:,}")
    print(f"   Quality Score: {result['incidents']['quality_score']:.1f}%")
