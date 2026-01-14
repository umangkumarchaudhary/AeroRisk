"""
AeroRisk - Data Validators
Data quality validation and integrity checks
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    check_name: str
    records_checked: int
    records_passed: int
    records_failed: int
    error_details: List[Dict] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        if self.records_checked == 0:
            return 0.0
        return self.records_passed / self.records_checked * 100


@dataclass
class DataQualityReport:
    """Overall data quality report."""
    source: str
    timestamp: datetime
    total_records: int
    validations: List[ValidationResult]
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    accuracy_score: float = 0.0
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'total_records': self.total_records,
            'completeness_score': self.completeness_score,
            'consistency_score': self.consistency_score,
            'accuracy_score': self.accuracy_score,
            'overall_score': self.overall_score,
            'validations': [
                {
                    'check_name': v.check_name,
                    'is_valid': v.is_valid,
                    'pass_rate': v.pass_rate,
                    'records_failed': v.records_failed
                }
                for v in self.validations
            ]
        }


class IncidentValidator:
    """Validate incident data quality."""
    
    # Required fields for incident records
    REQUIRED_FIELDS = ['source', 'incident_date', 'severity']
    
    # Valid enum values
    VALID_SOURCES = ['NTSB', 'ASRS', 'FAA', 'SYNTHETIC', 'INTERNAL']
    VALID_SEVERITIES = ['NONE', 'MINOR', 'SERIOUS', 'FATAL']
    VALID_PHASES = [
        'PREFLIGHT', 'TAXI', 'TAKEOFF', 'CLIMB', 'CRUISE',
        'DESCENT', 'APPROACH', 'LANDING', 'POST_FLIGHT', 'UNKNOWN'
    ]
    
    def __init__(self):
        self.validations: List[ValidationResult] = []
    
    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """Run all validations on incident data."""
        logger.info(f"Validating {len(df):,} incident records...")
        
        self.validations = []
        
        # Run all checks
        self._check_required_fields(df)
        self._check_source_values(df)
        self._check_severity_values(df)
        self._check_date_validity(df)
        self._check_injury_consistency(df)
        self._check_coordinate_validity(df)
        self._check_phase_values(df)
        
        # Calculate scores
        completeness = self._calculate_completeness(df)
        consistency = self._calculate_consistency()
        accuracy = self._calculate_accuracy()
        overall = (completeness * 0.3 + consistency * 0.4 + accuracy * 0.3)
        
        report = DataQualityReport(
            source='incidents',
            timestamp=datetime.now(),
            total_records=len(df),
            validations=self.validations,
            completeness_score=completeness,
            consistency_score=consistency,
            accuracy_score=accuracy,
            overall_score=overall
        )
        
        logger.info(f"Validation complete. Overall score: {overall:.1f}%")
        return report
    
    def _check_required_fields(self, df: pd.DataFrame) -> None:
        """Check that required fields are present and non-null."""
        for field in self.REQUIRED_FIELDS:
            if field in df.columns:
                null_count = df[field].isna().sum()
                passed = len(df) - null_count
                is_valid = null_count == 0
            else:
                null_count = len(df)
                passed = 0
                is_valid = False
            
            self.validations.append(ValidationResult(
                is_valid=is_valid,
                check_name=f"required_field_{field}",
                records_checked=len(df),
                records_passed=passed,
                records_failed=null_count
            ))
    
    def _check_source_values(self, df: pd.DataFrame) -> None:
        """Check source field contains valid values."""
        if 'source' not in df.columns:
            return
        
        invalid_mask = ~df['source'].isin(self.VALID_SOURCES)
        invalid_count = invalid_mask.sum()
        
        self.validations.append(ValidationResult(
            is_valid=invalid_count == 0,
            check_name="valid_source",
            records_checked=len(df),
            records_passed=len(df) - invalid_count,
            records_failed=invalid_count
        ))
    
    def _check_severity_values(self, df: pd.DataFrame) -> None:
        """Check severity field contains valid values."""
        if 'severity' not in df.columns:
            return
        
        invalid_mask = ~df['severity'].isin(self.VALID_SEVERITIES)
        invalid_count = invalid_mask.sum()
        
        self.validations.append(ValidationResult(
            is_valid=invalid_count == 0,
            check_name="valid_severity",
            records_checked=len(df),
            records_passed=len(df) - invalid_count,
            records_failed=invalid_count
        ))
    
    def _check_date_validity(self, df: pd.DataFrame) -> None:
        """Check incident dates are valid and reasonable."""
        if 'incident_date' not in df.columns:
            return
        
        dates = pd.to_datetime(df['incident_date'], errors='coerce')
        
        # Check for null dates
        null_dates = dates.isna().sum()
        
        # Check for future dates
        future_dates = (dates > datetime.now()).sum()
        
        # Check for very old dates (before 1950)
        old_dates = (dates < datetime(1950, 1, 1)).sum()
        
        invalid_count = null_dates + future_dates + old_dates
        
        self.validations.append(ValidationResult(
            is_valid=invalid_count == 0,
            check_name="valid_dates",
            records_checked=len(df),
            records_passed=len(df) - invalid_count,
            records_failed=invalid_count
        ))
    
    def _check_injury_consistency(self, df: pd.DataFrame) -> None:
        """Check injury counts are consistent with severity."""
        injury_cols = ['injuries_fatal', 'injuries_serious', 'injuries_minor']
        
        if not all(col in df.columns for col in injury_cols):
            return
        
        # Fatal severity should have fatal injuries
        if 'severity' in df.columns:
            fatal_mask = df['severity'] == 'FATAL'
            fatal_no_injuries = (
                fatal_mask & (df['injuries_fatal'].fillna(0) == 0)
            ).sum()
            
            self.validations.append(ValidationResult(
                is_valid=fatal_no_injuries == 0,
                check_name="injury_severity_consistency",
                records_checked=fatal_mask.sum(),
                records_passed=fatal_mask.sum() - fatal_no_injuries,
                records_failed=fatal_no_injuries
            ))
        
        # Injury counts should be non-negative
        for col in injury_cols:
            negative_count = (df[col].fillna(0) < 0).sum()
            self.validations.append(ValidationResult(
                is_valid=negative_count == 0,
                check_name=f"non_negative_{col}",
                records_checked=len(df),
                records_passed=len(df) - negative_count,
                records_failed=negative_count
            ))
    
    def _check_coordinate_validity(self, df: pd.DataFrame) -> None:
        """Check latitude/longitude are in valid ranges."""
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return
        
        lat_mask = df['latitude'].notna()
        lon_mask = df['longitude'].notna()
        coord_mask = lat_mask & lon_mask
        
        if coord_mask.sum() == 0:
            return
        
        # Valid ranges: lat -90 to 90, lon -180 to 180
        invalid_lat = (
            (df['latitude'] < -90) | (df['latitude'] > 90)
        ) & lat_mask
        
        invalid_lon = (
            (df['longitude'] < -180) | (df['longitude'] > 180)
        ) & lon_mask
        
        invalid_count = (invalid_lat | invalid_lon).sum()
        
        self.validations.append(ValidationResult(
            is_valid=invalid_count == 0,
            check_name="valid_coordinates",
            records_checked=coord_mask.sum(),
            records_passed=coord_mask.sum() - invalid_count,
            records_failed=invalid_count
        ))
    
    def _check_phase_values(self, df: pd.DataFrame) -> None:
        """Check flight phase contains valid values."""
        if 'phase_of_flight' not in df.columns:
            return
        
        # Only check non-null values
        non_null_mask = df['phase_of_flight'].notna()
        invalid_mask = (
            non_null_mask & 
            ~df['phase_of_flight'].isin(self.VALID_PHASES)
        )
        invalid_count = invalid_mask.sum()
        
        self.validations.append(ValidationResult(
            is_valid=invalid_count == 0,
            check_name="valid_flight_phase",
            records_checked=non_null_mask.sum(),
            records_passed=non_null_mask.sum() - invalid_count,
            records_failed=invalid_count
        ))
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        important_fields = [
            'source', 'incident_date', 'severity', 'location',
            'aircraft_make', 'phase_of_flight'
        ]
        
        completeness_scores = []
        for field in important_fields:
            if field in df.columns:
                score = (1 - df[field].isna().mean()) * 100
                completeness_scores.append(score)
        
        return np.mean(completeness_scores) if completeness_scores else 0
    
    def _calculate_consistency(self) -> float:
        """Calculate consistency score from validation results."""
        if not self.validations:
            return 0
        
        return np.mean([v.pass_rate for v in self.validations])
    
    def _calculate_accuracy(self) -> float:
        """Calculate accuracy score (based on enum validations)."""
        enum_checks = [
            v for v in self.validations 
            if v.check_name.startswith('valid_')
        ]
        
        if not enum_checks:
            return 100
        
        return np.mean([v.pass_rate for v in enum_checks])


class OperationalValidator:
    """Validate operational data quality."""
    
    def __init__(self):
        self.validations: List[ValidationResult] = []
    
    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """Run all validations on operational data."""
        logger.info(f"Validating {len(df):,} operational records...")
        
        self.validations = []
        
        # Run checks
        self._check_required_fields(df)
        self._check_numeric_ranges(df)
        self._check_date_validity(df)
        
        # Calculate scores
        completeness = self._calculate_completeness(df)
        consistency = np.mean([v.pass_rate for v in self.validations]) if self.validations else 0
        
        report = DataQualityReport(
            source='operational_data',
            timestamp=datetime.now(),
            total_records=len(df),
            validations=self.validations,
            completeness_score=completeness,
            consistency_score=consistency,
            accuracy_score=consistency,
            overall_score=(completeness + consistency * 2) / 3
        )
        
        logger.info(f"Validation complete. Overall score: {report.overall_score:.1f}%")
        return report
    
    def _check_required_fields(self, df: pd.DataFrame) -> None:
        """Check required fields are present."""
        required = ['date', 'aircraft_id']
        
        for field in required:
            if field in df.columns:
                null_count = df[field].isna().sum()
                passed = len(df) - null_count
            else:
                null_count = len(df)
                passed = 0
            
            self.validations.append(ValidationResult(
                is_valid=null_count == 0,
                check_name=f"required_{field}",
                records_checked=len(df),
                records_passed=passed,
                records_failed=null_count
            ))
    
    def _check_numeric_ranges(self, df: pd.DataFrame) -> None:
        """Check numeric fields are in valid ranges."""
        range_checks = {
            'crew_duty_hours': (0, 24),
            'crew_rest_hours': (0, 48),
            'crew_fatigue_index': (0, 100),
            'aircraft_age_years': (0, 100),
            'operational_risk_score': (0, 100)
        }
        
        for field, (min_val, max_val) in range_checks.items():
            if field not in df.columns:
                continue
            
            valid_mask = df[field].isna() | (
                (df[field] >= min_val) & (df[field] <= max_val)
            )
            invalid_count = (~valid_mask).sum()
            
            self.validations.append(ValidationResult(
                is_valid=invalid_count == 0,
                check_name=f"range_{field}",
                records_checked=len(df),
                records_passed=len(df) - invalid_count,
                records_failed=invalid_count
            ))
    
    def _check_date_validity(self, df: pd.DataFrame) -> None:
        """Check dates are valid."""
        if 'date' not in df.columns:
            return
        
        dates = pd.to_datetime(df['date'], errors='coerce')
        invalid_count = dates.isna().sum()
        
        self.validations.append(ValidationResult(
            is_valid=invalid_count == 0,
            check_name="valid_date",
            records_checked=len(df),
            records_passed=len(df) - invalid_count,
            records_failed=invalid_count
        ))
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate completeness score."""
        important_fields = [
            'date', 'aircraft_id', 'crew_duty_hours',
            'aircraft_age_years', 'operational_risk_score'
        ]
        
        scores = []
        for field in important_fields:
            if field in df.columns:
                score = (1 - df[field].isna().mean()) * 100
                scores.append(score)
        
        return np.mean(scores) if scores else 0
