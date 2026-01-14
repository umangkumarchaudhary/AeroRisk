"""
AeroRisk - Predictive Safety Risk Analytics Platform
SQLAlchemy ORM Models

This module defines the database schema for the AeroRisk platform,
aligned with the Safety Management System (SMS) framework.
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, List

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Date,
    ForeignKey, Text, Enum, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.sql import func


# ============================================
# Base Class
# ============================================

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# ============================================
# Enums
# ============================================

class DataSource(str, PyEnum):
    """Source of incident data."""
    NTSB = "NTSB"
    ASRS = "ASRS"
    FAA = "FAA"
    SYNTHETIC = "SYNTHETIC"
    INTERNAL = "INTERNAL"


class SeverityLevel(str, PyEnum):
    """Incident severity classification."""
    NONE = "NONE"
    MINOR = "MINOR"
    SERIOUS = "SERIOUS"
    FATAL = "FATAL"


class RiskLevel(str, PyEnum):
    """Risk level classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FlightPhase(str, PyEnum):
    """Phase of flight when incident occurred."""
    PREFLIGHT = "PREFLIGHT"
    TAXI = "TAXI"
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    DESCENT = "DESCENT"
    APPROACH = "APPROACH"
    LANDING = "LANDING"
    POST_FLIGHT = "POST_FLIGHT"
    UNKNOWN = "UNKNOWN"


class SMSPillar(str, PyEnum):
    """SMS (Safety Management System) pillars."""
    SAFETY_POLICY = "SAFETY_POLICY"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    SAFETY_ASSURANCE = "SAFETY_ASSURANCE"
    SAFETY_PROMOTION = "SAFETY_PROMOTION"


class RecommendationStatus(str, PyEnum):
    """Status of prescriptive recommendations."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    IMPLEMENTED = "IMPLEMENTED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"


class RecommendationPriority(str, PyEnum):
    """Priority level for recommendations."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================
# Core Tables
# ============================================

class Incident(Base):
    """
    Historical incident records from multiple sources.
    Core table for predictive analytics.
    """
    __tablename__ = "incidents"
    __table_args__ = (
        Index("idx_incidents_date", "incident_date"),
        Index("idx_incidents_severity", "severity"),
        Index("idx_incidents_source", "source"),
        Index("idx_incidents_location", "location"),
        {"schema": "ingestion"}
    )
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Source identification
    source: Mapped[str] = mapped_column(
        Enum(DataSource), 
        nullable=False,
        index=True
    )
    external_id: Mapped[Optional[str]] = mapped_column(String(100))  # Original ID from source
    
    # Incident details
    incident_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    report_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Location
    location: Mapped[Optional[str]] = mapped_column(String(255))
    airport_code: Mapped[Optional[str]] = mapped_column(String(10))
    country: Mapped[Optional[str]] = mapped_column(String(100))
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    
    # Aircraft
    aircraft_type: Mapped[Optional[str]] = mapped_column(String(100))
    aircraft_make: Mapped[Optional[str]] = mapped_column(String(100))
    aircraft_model: Mapped[Optional[str]] = mapped_column(String(100))
    aircraft_registration: Mapped[Optional[str]] = mapped_column(String(20))
    operator: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Flight details
    phase_of_flight: Mapped[Optional[str]] = mapped_column(Enum(FlightPhase))
    flight_type: Mapped[Optional[str]] = mapped_column(String(50))  # Commercial, Private, Cargo, etc.
    
    # Severity and outcomes
    severity: Mapped[str] = mapped_column(
        Enum(SeverityLevel), 
        default=SeverityLevel.NONE,
        nullable=False
    )
    injuries_fatal: Mapped[int] = mapped_column(Integer, default=0)
    injuries_serious: Mapped[int] = mapped_column(Integer, default=0)
    injuries_minor: Mapped[int] = mapped_column(Integer, default=0)
    injuries_uninjured: Mapped[int] = mapped_column(Integer, default=0)
    aircraft_damage: Mapped[Optional[str]] = mapped_column(String(50))  # None, Minor, Substantial, Destroyed
    
    # Cause analysis
    probable_cause: Mapped[Optional[str]] = mapped_column(Text)
    contributing_factors: Mapped[Optional[dict]] = mapped_column(JSONB)
    event_type: Mapped[Optional[str]] = mapped_column(String(100))  # Accident, Incident, etc.
    
    # Weather at time of incident
    weather_conditions: Mapped[Optional[str]] = mapped_column(String(50))  # VMC, IMC
    weather_data: Mapped[Optional[dict]] = mapped_column(JSONB)  # Detailed weather info
    
    # Metadata
    raw_data: Mapped[Optional[dict]] = mapped_column(JSONB)  # Original source data
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    risk_predictions: Mapped[List["RiskPrediction"]] = relationship(back_populates="incident")


class WeatherCondition(Base):
    """
    Weather data for airports/locations.
    Used for risk correlation analysis.
    """
    __tablename__ = "weather_conditions"
    __table_args__ = (
        Index("idx_weather_date_airport", "date", "airport_code"),
        UniqueConstraint("date", "airport_code", "hour", name="uq_weather_datetime_airport"),
        {"schema": "ingestion"}
    )
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Location and time
    date: Mapped[datetime] = mapped_column(Date, nullable=False)
    hour: Mapped[int] = mapped_column(Integer, default=0)  # 0-23
    airport_code: Mapped[str] = mapped_column(String(10), nullable=False)
    
    # Weather parameters
    temperature_c: Mapped[Optional[float]] = mapped_column(Float)
    feels_like_c: Mapped[Optional[float]] = mapped_column(Float)
    humidity_percent: Mapped[Optional[float]] = mapped_column(Float)
    pressure_hpa: Mapped[Optional[float]] = mapped_column(Float)
    
    # Visibility and clouds
    visibility_m: Mapped[Optional[float]] = mapped_column(Float)
    cloud_cover_percent: Mapped[Optional[float]] = mapped_column(Float)
    ceiling_ft: Mapped[Optional[float]] = mapped_column(Float)
    
    # Wind
    wind_speed_kt: Mapped[Optional[float]] = mapped_column(Float)
    wind_gust_kt: Mapped[Optional[float]] = mapped_column(Float)
    wind_direction_deg: Mapped[Optional[float]] = mapped_column(Float)
    
    # Precipitation
    precipitation_mm: Mapped[Optional[float]] = mapped_column(Float)
    precipitation_type: Mapped[Optional[str]] = mapped_column(String(50))  # Rain, Snow, Sleet, etc.
    
    # Conditions
    weather_condition: Mapped[Optional[str]] = mapped_column(String(100))  # Clear, Cloudy, Fog, etc.
    severe_weather_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Calculated risk score for weather
    weather_risk_score: Mapped[Optional[float]] = mapped_column(Float)  # 0-100
    
    # Metadata
    source: Mapped[Optional[str]] = mapped_column(String(50))  # OpenWeather, NOAA, etc.
    raw_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class OperationalData(Base):
    """
    Synthetic operational data for training.
    Includes crew, maintenance, and schedule information.
    """
    __tablename__ = "operational_data"
    __table_args__ = (
        Index("idx_ops_date", "date"),
        Index("idx_ops_aircraft", "aircraft_id"),
        {"schema": "ingestion"}
    )
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Identifiers
    date: Mapped[datetime] = mapped_column(Date, nullable=False)
    flight_id: Mapped[Optional[str]] = mapped_column(String(20))
    aircraft_id: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Route
    origin_airport: Mapped[Optional[str]] = mapped_column(String(10))
    destination_airport: Mapped[Optional[str]] = mapped_column(String(10))
    scheduled_departure: Mapped[Optional[datetime]] = mapped_column(DateTime)
    actual_departure: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Crew factors
    crew_duty_hours: Mapped[Optional[float]] = mapped_column(Float)
    crew_rest_hours: Mapped[Optional[float]] = mapped_column(Float)
    crew_fatigue_index: Mapped[Optional[float]] = mapped_column(Float)  # 0-100, higher = more fatigued
    pilot_experience_hours: Mapped[Optional[float]] = mapped_column(Float)
    
    # Maintenance
    aircraft_age_years: Mapped[Optional[float]] = mapped_column(Float)
    days_since_major_maintenance: Mapped[Optional[int]] = mapped_column(Integer)
    maintenance_overdue_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    open_maintenance_items: Mapped[int] = mapped_column(Integer, default=0)
    
    # Schedule
    schedule_deviation_minutes: Mapped[Optional[int]] = mapped_column(Integer)
    turnaround_time_minutes: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Calculated risk factors
    operational_risk_score: Mapped[Optional[float]] = mapped_column(Float)  # 0-100
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


# ============================================
# ML & Analytics Tables
# ============================================

class RiskPrediction(Base):
    """
    Model predictions for risk scores.
    Links predictions to incidents and recommendations.
    """
    __tablename__ = "risk_predictions"
    __table_args__ = (
        Index("idx_predictions_date", "prediction_date"),
        Index("idx_predictions_risk_level", "risk_level"),
        {"schema": "ml"}
    )
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Prediction target
    prediction_date: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # FLIGHT, AIRPORT, ROUTE, AIRCRAFT
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Prediction output
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0-100
    risk_level: Mapped[str] = mapped_column(Enum(RiskLevel), nullable=False)
    severity_prediction: Mapped[Optional[str]] = mapped_column(Enum(SeverityLevel))
    
    # Model confidence
    confidence: Mapped[float] = mapped_column(Float, default=0.0)  # 0-1
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Feature importance (SHAP values)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSONB)
    features_used: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Linked incident (if this is a post-hoc analysis)
    incident_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("ingestion.incidents.id")
    )
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    # Relationships
    incident: Mapped[Optional["Incident"]] = relationship(back_populates="risk_predictions")
    recommendations: Mapped[List["Recommendation"]] = relationship(back_populates="prediction")


class Recommendation(Base):
    """
    Prescriptive recommendations from the analytics engine.
    Contains actionable insights with ROI analysis.
    """
    __tablename__ = "recommendations"
    __table_args__ = (
        Index("idx_recommendations_status", "status"),
        Index("idx_recommendations_priority", "priority"),
        {"schema": "analytics"}
    )
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Link to prediction
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("ml.risk_predictions.id"),
        nullable=False
    )
    
    # Recommendation details
    action_type: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Impact analysis
    expected_risk_reduction: Mapped[float] = mapped_column(Float)  # Percentage
    implementation_cost: Mapped[Optional[str]] = mapped_column(String(50))  # $, $$, $$$, $$$$
    implementation_cost_value: Mapped[Optional[float]] = mapped_column(Float)  # Actual dollar amount
    roi_score: Mapped[Optional[float]] = mapped_column(Float)  # Calculated ROI
    
    # Prioritization
    priority: Mapped[str] = mapped_column(
        Enum(RecommendationPriority), 
        default=RecommendationPriority.MEDIUM
    )
    
    # Implementation
    status: Mapped[str] = mapped_column(
        Enum(RecommendationStatus), 
        default=RecommendationStatus.PENDING
    )
    implementation_notes: Mapped[Optional[str]] = mapped_column(Text)
    implemented_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    implemented_by: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    prediction: Mapped["RiskPrediction"] = relationship(back_populates="recommendations")


class SafetyKPI(Base):
    """
    Safety Performance Indicators (SPIs) and Targets (SPTs).
    Aligned with SMS framework pillars.
    """
    __tablename__ = "safety_kpis"
    __table_args__ = (
        Index("idx_kpis_date", "measurement_date"),
        Index("idx_kpis_type", "kpi_type"),
        Index("idx_kpis_pillar", "sms_pillar"),
        {"schema": "analytics"}
    )
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # KPI identification
    kpi_type: Mapped[str] = mapped_column(String(10), nullable=False)  # SPI or SPT
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    code: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., SPI-001
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Measurement
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[Optional[str]] = mapped_column(String(50))  # e.g., "per 1000 flights"
    measurement_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    measurement_period: Mapped[Optional[str]] = mapped_column(String(20))  # daily, weekly, monthly
    
    # Targets
    target_value: Mapped[Optional[float]] = mapped_column(Float)
    threshold_warning: Mapped[Optional[float]] = mapped_column(Float)
    threshold_critical: Mapped[Optional[float]] = mapped_column(Float)
    
    # Status
    target_met: Mapped[Optional[bool]] = mapped_column(Boolean)
    trend_direction: Mapped[Optional[str]] = mapped_column(String(20))  # improving, stable, declining
    
    # SMS alignment
    sms_pillar: Mapped[str] = mapped_column(Enum(SMSPillar), nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(100))  # e.g., "Flight Operations", "Maintenance"
    
    # Leading/Lagging indicator
    indicator_type: Mapped[str] = mapped_column(String(20), default="lagging")  # leading, lagging
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class ModelRegistry(Base):
    """
    Registry for ML model versions and metadata.
    Tracks model performance and deployment status.
    """
    __tablename__ = "model_registry"
    __table_args__ = {"schema": "ml"}
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Model identification
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # xgboost, lightgbm, etc.
    
    # Model artifact
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Performance metrics
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    precision: Mapped[Optional[float]] = mapped_column(Float)
    recall: Mapped[Optional[float]] = mapped_column(Float)
    f1_score: Mapped[Optional[float]] = mapped_column(Float)
    auc_roc: Mapped[Optional[float]] = mapped_column(Float)
    
    # Additional metrics (stored as JSON)
    metrics: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Training info
    training_data_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    training_data_end: Mapped[Optional[datetime]] = mapped_column(DateTime)
    training_samples: Mapped[Optional[int]] = mapped_column(Integer)
    features: Mapped[Optional[list]] = mapped_column(ARRAY(String))
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Deployment
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    deployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    created_by: Mapped[Optional[str]] = mapped_column(String(255))


# ============================================
# Audit & Logging Tables
# ============================================

class AuditLog(Base):
    """
    Audit trail for all system actions.
    Important for compliance and debugging.
    """
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("idx_audit_timestamp", "timestamp"),
        Index("idx_audit_action", "action"),
        {"schema": "analytics"}
    )
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[Optional[str]] = mapped_column(String(100))
    entity_id: Mapped[Optional[str]] = mapped_column(String(100))
    user: Mapped[Optional[str]] = mapped_column(String(255))
    details: Mapped[Optional[dict]] = mapped_column(JSONB)
    ip_address: Mapped[Optional[str]] = mapped_column(String(50))


class DataQualityLog(Base):
    """
    Data quality metrics and validation results.
    Tracks data integrity over time.
    """
    __tablename__ = "data_quality_logs"
    __table_args__ = {"schema": "analytics"}
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    run_date: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Metrics
    total_records: Mapped[int] = mapped_column(Integer, default=0)
    valid_records: Mapped[int] = mapped_column(Integer, default=0)
    invalid_records: Mapped[int] = mapped_column(Integer, default=0)
    
    # Quality scores
    completeness_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0-100
    consistency_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0-100
    accuracy_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0-100
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0-100
    
    # Details
    validation_errors: Mapped[Optional[dict]] = mapped_column(JSONB)
    recommendations: Mapped[Optional[list]] = mapped_column(ARRAY(String))
