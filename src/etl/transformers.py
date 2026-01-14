"""
AeroRisk - Data Transformers
Feature engineering and data transformation for ML models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from loguru import logger


class IncidentTransformer:
    """Transform raw incident data into ML-ready features."""
    
    def __init__(self):
        self.severity_weights = {
            'NONE': 0,
            'MINOR': 1,
            'SERIOUS': 3,
            'FATAL': 5
        }
        
        self.phase_risk_weights = {
            'TAKEOFF': 1.5,
            'LANDING': 1.4,
            'APPROACH': 1.3,
            'CLIMB': 1.1,
            'DESCENT': 1.1,
            'CRUISE': 0.8,
            'TAXI': 0.7,
            'PREFLIGHT': 0.5,
            'POST_FLIGHT': 0.5,
            'UNKNOWN': 1.0
        }
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations to incident data."""
        logger.info(f"Transforming {len(df):,} incidents...")
        
        result = df.copy()
        
        # Date features
        result = self._extract_date_features(result)
        
        # Severity encoding
        result = self._encode_severity(result)
        
        # Flight phase risk
        result = self._encode_flight_phase(result)
        
        # Injury aggregation
        result = self._aggregate_injuries(result)
        
        # Location features
        result = self._process_location(result)
        
        # Weather risk score
        result = self._calculate_weather_risk(result)
        
        logger.info(f"Transformation complete. Shape: {result.shape}")
        return result
    
    def _extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from incident_date."""
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        
        df['year'] = df['incident_date'].dt.year
        df['month'] = df['incident_date'].dt.month
        df['day_of_week'] = df['incident_date'].dt.dayofweek
        df['day_of_year'] = df['incident_date'].dt.dayofyear
        df['quarter'] = df['incident_date'].dt.quarter
        
        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Holiday proximity (simplified - just major US holidays)
        df['is_holiday_season'] = df['month'].isin([11, 12, 7]).astype(int)
        
        return df
    
    def _encode_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode severity as numeric values."""
        df['severity_code'] = df['severity'].map(self.severity_weights).fillna(0)
        
        # Binary target: serious incident (SERIOUS or FATAL)
        df['is_serious'] = (df['severity'].isin(['SERIOUS', 'FATAL'])).astype(int)
        
        return df
    
    def _encode_flight_phase(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add flight phase risk factor."""
        df['phase_risk'] = df['phase_of_flight'].map(self.phase_risk_weights).fillna(1.0)
        return df
    
    def _aggregate_injuries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate total injuries and injury rate."""
        df['total_injuries'] = (
            df['injuries_fatal'].fillna(0) + 
            df['injuries_serious'].fillna(0) + 
            df['injuries_minor'].fillna(0)
        )
        
        df['total_people'] = df['total_injuries'] + df['injuries_uninjured'].fillna(0)
        
        df['injury_rate'] = np.where(
            df['total_people'] > 0,
            df['total_injuries'] / df['total_people'],
            0
        )
        
        # Weighted injury score
        df['injury_score'] = (
            df['injuries_fatal'].fillna(0) * 5 +
            df['injuries_serious'].fillna(0) * 3 +
            df['injuries_minor'].fillna(0) * 1
        )
        
        return df
    
    def _process_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process location data."""
        # Has coordinates flag
        df['has_coordinates'] = (
            df['latitude'].notna() & df['longitude'].notna()
        ).astype(int)
        
        # US vs international
        df['is_us'] = (df['country'] == 'USA').astype(int)
        
        return df
    
    def _calculate_weather_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weather-based risk score."""
        weather_risk_map = {
            'VMC': 0.3,  # Visual Meteorological Conditions - good
            'IMC': 0.8,  # Instrument Meteorological Conditions - poor
        }
        
        df['weather_risk'] = df['weather_conditions'].map(weather_risk_map).fillna(0.5)
        return df


class OperationalTransformer:
    """Transform operational data for ML features."""
    
    def __init__(self):
        pass
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations to operational data."""
        logger.info(f"Transforming {len(df):,} operational records...")
        
        result = df.copy()
        
        # Crew fatigue risk
        result = self._calculate_fatigue_risk(result)
        
        # Maintenance risk
        result = self._calculate_maintenance_risk(result)
        
        # Schedule risk
        result = self._calculate_schedule_risk(result)
        
        # Combined operational risk
        result = self._calculate_combined_risk(result)
        
        logger.info(f"Transformation complete. Shape: {result.shape}")
        return result
    
    def _calculate_fatigue_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate crew fatigue risk score."""
        # Higher duty hours, lower rest = higher risk
        df['fatigue_risk'] = np.clip(
            (df['crew_duty_hours'].fillna(8) / 16) * 
            (1 - df['crew_rest_hours'].fillna(8) / 12),
            0, 1
        ) * 100
        
        return df
    
    def _calculate_maintenance_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate maintenance-based risk score."""
        # Overdue maintenance and open items increase risk
        overdue_factor = df['maintenance_overdue_flag'].fillna(0) * 30
        items_factor = np.clip(df['open_maintenance_items'].fillna(0) * 5, 0, 40)
        age_factor = np.clip(df['aircraft_age_years'].fillna(10) * 1, 0, 30)
        
        df['maintenance_risk'] = overdue_factor + items_factor + age_factor
        
        return df
    
    def _calculate_schedule_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate schedule-based risk score."""
        # Rushed turnarounds and delays increase risk
        deviation_factor = np.clip(
            np.abs(df['schedule_deviation_minutes'].fillna(0)) / 60 * 10, 0, 30
        )
        
        turnaround_factor = np.where(
            df['turnaround_time_minutes'].fillna(60) < 30,
            20,
            0
        )
        
        df['schedule_risk'] = deviation_factor + turnaround_factor
        
        return df
    
    def _calculate_combined_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate combined operational risk score."""
        df['operational_risk_score'] = (
            df['fatigue_risk'] * 0.4 +
            df['maintenance_risk'] * 0.4 +
            df['schedule_risk'] * 0.2
        )
        
        return df


class FeatureEngineer:
    """Combined feature engineering for ML models."""
    
    def __init__(self):
        self.incident_transformer = IncidentTransformer()
        self.operational_transformer = OperationalTransformer()
    
    def engineer_incident_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML-ready features from incident data."""
        return self.incident_transformer.transform(df)
    
    def engineer_operational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML-ready features from operational data."""
        return self.operational_transformer.transform(df)
    
    def create_training_dataset(
        self, 
        incidents_df: pd.DataFrame, 
        operations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create combined training dataset from incidents and operations."""
        logger.info("Creating combined training dataset...")
        
        # Transform both
        incidents_features = self.engineer_incident_features(incidents_df)
        operations_features = self.engineer_operational_features(operations_df)
        
        # For training, we'll aggregate historical incident patterns
        # and combine with operational features
        
        # Aggregate incidents by time period
        incident_agg = incidents_features.groupby(['year', 'month']).agg({
            'severity_code': ['mean', 'max', 'count'],
            'is_serious': 'sum',
            'injury_score': 'sum',
            'phase_risk': 'mean',
            'weather_risk': 'mean'
        }).reset_index()
        
        incident_agg.columns = [
            'year', 'month',
            'avg_severity', 'max_severity', 'incident_count',
            'serious_count', 'total_injury_score',
            'avg_phase_risk', 'avg_weather_risk'
        ]
        
        logger.info(f"Created aggregated incident features: {incident_agg.shape}")
        logger.info(f"Operations features: {operations_features.shape}")
        
        return incidents_features, operations_features, incident_agg
