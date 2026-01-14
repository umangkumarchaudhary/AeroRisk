"""
AeroRisk Database Module
"""

from src.database.connection import (
    engine,
    SessionLocal,
    get_db,
    get_db_session,
    verify_connection,
    get_database_info,
    db_config,
)

from src.database.models import (
    Base,
    Incident,
    WeatherCondition,
    OperationalData,
    RiskPrediction,
    Recommendation,
    SafetyKPI,
    ModelRegistry,
    AuditLog,
    DataQualityLog,
    # Enums
    DataSource,
    SeverityLevel,
    RiskLevel,
    FlightPhase,
    SMSPillar,
    RecommendationStatus,
    RecommendationPriority,
)

__all__ = [
    # Connection
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_session",
    "verify_connection",
    "get_database_info",
    "db_config",
    # Base
    "Base",
    # Models
    "Incident",
    "WeatherCondition",
    "OperationalData",
    "RiskPrediction",
    "Recommendation",
    "SafetyKPI",
    "ModelRegistry",
    "AuditLog",
    "DataQualityLog",
    # Enums
    "DataSource",
    "SeverityLevel",
    "RiskLevel",
    "FlightPhase",
    "SMSPillar",
    "RecommendationStatus",
    "RecommendationPriority",
]
