"""
AeroRisk - Pydantic Schemas
============================
Data validation schemas for API requests and responses.

Author: Umang Kumar
Date: 2024-01-15
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime, date
from enum import Enum


# ============================================
# Enums
# ============================================

class SeverityLevel(str, Enum):
    NONE = "NONE"
    MINOR = "MINOR"
    SERIOUS = "SERIOUS"
    FATAL = "FATAL"


class Priority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FlightPhase(str, Enum):
    TAXI = "TAXI"
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    DESCENT = "DESCENT"
    APPROACH = "APPROACH"
    LANDING = "LANDING"
    GO_AROUND = "GO_AROUND"


class WeatherCondition(str, Enum):
    VMC = "VMC"  # Visual Meteorological Conditions
    IMC = "IMC"  # Instrument Meteorological Conditions


# ============================================
# Request Schemas
# ============================================

class RiskPredictionRequest(BaseModel):
    """Request for risk prediction."""
    incident_year: int = Field(..., ge=1980, le=2030, description="Year of incident")
    incident_month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    incident_day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon)")
    is_weekend: int = Field(0, ge=0, le=1, description="Weekend flag")
    severity_code: int = Field(0, ge=0, le=3, description="Severity (0=NONE, 3=FATAL)")
    phase_risk_factor: float = Field(1.0, ge=0.5, le=2.0, description="Flight phase risk")
    total_injuries: int = Field(0, ge=0, description="Total injuries")
    injury_rate: float = Field(0.0, ge=0.0, description="Injury rate")
    has_coordinates: int = Field(1, ge=0, le=1, description="Has location data")
    is_us: int = Field(1, ge=0, le=1, description="US location")
    weather_risk_score: float = Field(0.0, ge=0.0, le=100.0, description="Weather risk 0-100")

    class Config:
        json_schema_extra = {
            "example": {
                "incident_year": 2024,
                "incident_month": 6,
                "incident_day_of_week": 2,
                "is_weekend": 0,
                "severity_code": 1,
                "phase_risk_factor": 1.2,
                "total_injuries": 0,
                "injury_rate": 0.0,
                "has_coordinates": 1,
                "is_us": 1,
                "weather_risk_score": 45.0
            }
        }


class SeverityClassificationRequest(BaseModel):
    """Request for severity classification."""
    weather_risk_score: float = Field(0.0, ge=0.0, le=100.0)
    crew_fatigue_index: float = Field(0.0, ge=0.0, le=100.0)
    maintenance_risk: float = Field(0.0, ge=0.0, le=100.0)
    aircraft_age_years: int = Field(10, ge=0, le=60)
    flight_phase: FlightPhase = Field(FlightPhase.CRUISE)
    visibility_m: float = Field(10000.0, ge=0.0)
    wind_speed_kt: float = Field(10.0, ge=0.0)
    is_night: bool = Field(False)
    crew_experience_hours: int = Field(5000, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "weather_risk_score": 65.0,
                "crew_fatigue_index": 40.0,
                "maintenance_risk": 30.0,
                "aircraft_age_years": 15,
                "flight_phase": "APPROACH",
                "visibility_m": 5000.0,
                "wind_speed_kt": 20.0,
                "is_night": False,
                "crew_experience_hours": 8000
            }
        }


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection."""
    operational_risk_score: float = Field(..., ge=0.0, le=100.0)
    fatigue_risk: float = Field(..., ge=0.0, le=100.0)
    maintenance_risk: float = Field(..., ge=0.0, le=100.0)
    schedule_risk: float = Field(0.0, ge=0.0, le=100.0)
    crew_duty_hours: float = Field(8.0, ge=0.0, le=20.0)
    crew_rest_hours: float = Field(12.0, ge=0.0, le=48.0)
    schedule_deviation_mins: float = Field(0.0)
    turnaround_time_mins: float = Field(45.0, ge=0.0)


class WhatIfScenarioRequest(BaseModel):
    """Request for what-if scenario analysis."""
    base_scenario: Dict[str, Any] = Field(..., description="Base scenario parameters")
    modifications: List[Dict[str, Any]] = Field(..., description="List of modifications to apply")
    
    class Config:
        json_schema_extra = {
            "example": {
                "base_scenario": {
                    "weather_risk_score": 75.0,
                    "crew_fatigue_index": 60.0,
                    "maintenance_risk": 40.0
                },
                "modifications": [
                    {"weather_risk_score": 30.0},
                    {"crew_fatigue_index": 20.0},
                    {"maintenance_risk": 10.0}
                ]
            }
        }


class RecommendationRequest(BaseModel):
    """Request for safety recommendations."""
    weather_risk_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    visibility_m: Optional[float] = Field(None, ge=0.0)
    wind_speed_kt: Optional[float] = Field(None, ge=0.0)
    crew_fatigue_index: Optional[float] = Field(None, ge=0.0, le=100.0)
    crew_duty_hours: Optional[float] = Field(None, ge=0.0, le=20.0)
    maintenance_risk: Optional[float] = Field(None, ge=0.0, le=100.0)
    maintenance_overdue_flag: Optional[int] = Field(None, ge=0, le=1)
    turnaround_time_mins: Optional[float] = Field(None, ge=0.0)
    injury_score: Optional[float] = Field(None, ge=0.0)
    severity_code: Optional[int] = Field(None, ge=0, le=3)


class AnalyticsQueryRequest(BaseModel):
    """Request for analytics queries."""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    severity: Optional[SeverityLevel] = None
    source: Optional[str] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


# ============================================
# Response Schemas
# ============================================

class RiskPredictionResponse(BaseModel):
    """Response from risk prediction."""
    risk_score: float = Field(..., description="Predicted risk score 0-100")
    risk_level: str = Field(..., description="Risk category")
    confidence: float = Field(..., description="Model confidence")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('risk_level', pre=True, always=True)
    def set_risk_level(cls, v, values):
        score = values.get('risk_score', 0)
        if score >= 75:
            return "CRITICAL"
        elif score >= 50:
            return "HIGH"
        elif score >= 25:
            return "MEDIUM"
        return "LOW"


class SeverityClassificationResponse(BaseModel):
    """Response from severity classification."""
    predicted_severity: SeverityLevel
    severity_code: int
    probabilities: Dict[str, float]
    confidence: float
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.now)


class AnomalyDetectionResponse(BaseModel):
    """Response from anomaly detection."""
    is_anomaly: bool
    anomaly_score: float = Field(..., description="Score (-1 to 0, more negative = more anomalous)")
    percentile: float = Field(..., description="Percentile rank")
    contributing_factors: List[str]
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.now)


class RecommendationItem(BaseModel):
    """A single recommendation."""
    id: str
    title: str
    description: str
    priority: Priority
    risk_reduction_percent: float
    cost_level: str
    implementation_time: str
    sms_pillar: str
    regulatory_refs: List[str]
    action_steps: List[str]
    kpis: List[str]


class RecommendationResponse(BaseModel):
    """Response from recommendation engine."""
    total_recommendations: int
    recommendations: List[RecommendationItem]
    total_risk_reduction: float
    timestamp: datetime = Field(default_factory=datetime.now)


class WhatIfScenarioResult(BaseModel):
    """Result for a single scenario."""
    scenario_name: str
    parameters: Dict[str, Any]
    risk_score: float
    risk_change: float
    risk_change_percent: float


class WhatIfScenarioResponse(BaseModel):
    """Response from what-if analysis."""
    base_risk_score: float
    scenarios: List[WhatIfScenarioResult]
    optimal_scenario: str
    max_risk_reduction: float
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================
# Analytics Response Schemas
# ============================================

class IncidentSummary(BaseModel):
    """Summary of incidents."""
    total_incidents: int
    by_severity: Dict[str, int]
    by_source: Dict[str, int]
    by_year: Dict[int, int]
    avg_injury_score: float
    fatal_rate: float


class TrendData(BaseModel):
    """Trend data point."""
    period: str
    count: int
    avg_risk: float
    fatal_count: int


class TrendAnalysis(BaseModel):
    """Trend analysis response."""
    metric: str
    period_type: str
    data: List[TrendData]
    trend_direction: str  # "increasing", "decreasing", "stable"
    change_percent: float


class RiskDistribution(BaseModel):
    """Risk distribution data."""
    category: str
    count: int
    percentage: float
    avg_risk_score: float


class AnalyticsDashboard(BaseModel):
    """Complete analytics dashboard data."""
    summary: IncidentSummary
    severity_distribution: List[RiskDistribution]
    monthly_trends: List[TrendData]
    top_risk_factors: List[Dict[str, Any]]
    recent_anomalies: List[Dict[str, Any]]
    model_performance: Dict[str, float]
    last_updated: datetime


class HealthCheckResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    database: str
    models_loaded: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)
