"""Initial schema with all core tables

Revision ID: 001
Revises: 
Create Date: 2026-01-14

This migration creates all initial tables for the AeroRisk platform:
- ingestion schema: incidents, weather_conditions, operational_data
- ml schema: risk_predictions, model_registry
- analytics schema: recommendations, safety_kpis, audit_logs, data_quality_logs
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all initial tables and schemas."""
    
    # Create schemas
    op.execute("CREATE SCHEMA IF NOT EXISTS ingestion")
    op.execute("CREATE SCHEMA IF NOT EXISTS ml")
    op.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    
    # Enable extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')
    
    # ==========================================
    # Ingestion Schema Tables
    # ==========================================
    
    # incidents table
    op.create_table(
        'incidents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('source', sa.String(20), nullable=False),
        sa.Column('external_id', sa.String(100), nullable=True),
        sa.Column('incident_date', sa.DateTime, nullable=False),
        sa.Column('report_date', sa.DateTime, nullable=True),
        sa.Column('location', sa.String(255), nullable=True),
        sa.Column('airport_code', sa.String(10), nullable=True),
        sa.Column('country', sa.String(100), nullable=True),
        sa.Column('latitude', sa.Float, nullable=True),
        sa.Column('longitude', sa.Float, nullable=True),
        sa.Column('aircraft_type', sa.String(100), nullable=True),
        sa.Column('aircraft_make', sa.String(100), nullable=True),
        sa.Column('aircraft_model', sa.String(100), nullable=True),
        sa.Column('aircraft_registration', sa.String(20), nullable=True),
        sa.Column('operator', sa.String(255), nullable=True),
        sa.Column('phase_of_flight', sa.String(20), nullable=True),
        sa.Column('flight_type', sa.String(50), nullable=True),
        sa.Column('severity', sa.String(20), nullable=False, server_default='NONE'),
        sa.Column('injuries_fatal', sa.Integer, server_default='0'),
        sa.Column('injuries_serious', sa.Integer, server_default='0'),
        sa.Column('injuries_minor', sa.Integer, server_default='0'),
        sa.Column('injuries_uninjured', sa.Integer, server_default='0'),
        sa.Column('aircraft_damage', sa.String(50), nullable=True),
        sa.Column('probable_cause', sa.Text, nullable=True),
        sa.Column('contributing_factors', postgresql.JSONB, nullable=True),
        sa.Column('event_type', sa.String(100), nullable=True),
        sa.Column('weather_conditions', sa.String(50), nullable=True),
        sa.Column('weather_data', postgresql.JSONB, nullable=True),
        sa.Column('raw_data', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('NOW()')),
        schema='ingestion'
    )
    
    op.create_index('idx_incidents_date', 'incidents', ['incident_date'], schema='ingestion')
    op.create_index('idx_incidents_severity', 'incidents', ['severity'], schema='ingestion')
    op.create_index('idx_incidents_source', 'incidents', ['source'], schema='ingestion')
    op.create_index('idx_incidents_location', 'incidents', ['location'], schema='ingestion')
    
    # weather_conditions table
    op.create_table(
        'weather_conditions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('date', sa.Date, nullable=False),
        sa.Column('hour', sa.Integer, server_default='0'),
        sa.Column('airport_code', sa.String(10), nullable=False),
        sa.Column('temperature_c', sa.Float, nullable=True),
        sa.Column('feels_like_c', sa.Float, nullable=True),
        sa.Column('humidity_percent', sa.Float, nullable=True),
        sa.Column('pressure_hpa', sa.Float, nullable=True),
        sa.Column('visibility_m', sa.Float, nullable=True),
        sa.Column('cloud_cover_percent', sa.Float, nullable=True),
        sa.Column('ceiling_ft', sa.Float, nullable=True),
        sa.Column('wind_speed_kt', sa.Float, nullable=True),
        sa.Column('wind_gust_kt', sa.Float, nullable=True),
        sa.Column('wind_direction_deg', sa.Float, nullable=True),
        sa.Column('precipitation_mm', sa.Float, nullable=True),
        sa.Column('precipitation_type', sa.String(50), nullable=True),
        sa.Column('weather_condition', sa.String(100), nullable=True),
        sa.Column('severe_weather_flag', sa.Boolean, server_default='false'),
        sa.Column('weather_risk_score', sa.Float, nullable=True),
        sa.Column('source', sa.String(50), nullable=True),
        sa.Column('raw_data', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        schema='ingestion'
    )
    
    op.create_index('idx_weather_date_airport', 'weather_conditions', ['date', 'airport_code'], schema='ingestion')
    op.create_unique_constraint('uq_weather_datetime_airport', 'weather_conditions', ['date', 'airport_code', 'hour'], schema='ingestion')
    
    # operational_data table
    op.create_table(
        'operational_data',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('date', sa.Date, nullable=False),
        sa.Column('flight_id', sa.String(20), nullable=True),
        sa.Column('aircraft_id', sa.String(20), nullable=False),
        sa.Column('origin_airport', sa.String(10), nullable=True),
        sa.Column('destination_airport', sa.String(10), nullable=True),
        sa.Column('scheduled_departure', sa.DateTime, nullable=True),
        sa.Column('actual_departure', sa.DateTime, nullable=True),
        sa.Column('crew_duty_hours', sa.Float, nullable=True),
        sa.Column('crew_rest_hours', sa.Float, nullable=True),
        sa.Column('crew_fatigue_index', sa.Float, nullable=True),
        sa.Column('pilot_experience_hours', sa.Float, nullable=True),
        sa.Column('aircraft_age_years', sa.Float, nullable=True),
        sa.Column('days_since_major_maintenance', sa.Integer, nullable=True),
        sa.Column('maintenance_overdue_flag', sa.Boolean, server_default='false'),
        sa.Column('open_maintenance_items', sa.Integer, server_default='0'),
        sa.Column('schedule_deviation_minutes', sa.Integer, nullable=True),
        sa.Column('turnaround_time_minutes', sa.Integer, nullable=True),
        sa.Column('operational_risk_score', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        schema='ingestion'
    )
    
    op.create_index('idx_ops_date', 'operational_data', ['date'], schema='ingestion')
    op.create_index('idx_ops_aircraft', 'operational_data', ['aircraft_id'], schema='ingestion')
    
    # ==========================================
    # ML Schema Tables
    # ==========================================
    
    # risk_predictions table
    op.create_table(
        'risk_predictions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('prediction_date', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('entity_type', sa.String(50), nullable=False),
        sa.Column('entity_id', sa.String(100), nullable=False),
        sa.Column('risk_score', sa.Float, nullable=False),
        sa.Column('risk_level', sa.String(20), nullable=False),
        sa.Column('severity_prediction', sa.String(20), nullable=True),
        sa.Column('confidence', sa.Float, server_default='0.0'),
        sa.Column('model_version', sa.String(50), nullable=False),
        sa.Column('feature_importance', postgresql.JSONB, nullable=True),
        sa.Column('features_used', postgresql.JSONB, nullable=True),
        sa.Column('incident_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['incident_id'], ['ingestion.incidents.id'], name='fk_predictions_incident'),
        schema='ml'
    )
    
    op.create_index('idx_predictions_date', 'risk_predictions', ['prediction_date'], schema='ml')
    op.create_index('idx_predictions_risk_level', 'risk_predictions', ['risk_level'], schema='ml')
    
    # model_registry table
    op.create_table(
        'model_registry',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('model_path', sa.String(500), nullable=False),
        sa.Column('accuracy', sa.Float, nullable=True),
        sa.Column('precision', sa.Float, nullable=True),
        sa.Column('recall', sa.Float, nullable=True),
        sa.Column('f1_score', sa.Float, nullable=True),
        sa.Column('auc_roc', sa.Float, nullable=True),
        sa.Column('metrics', postgresql.JSONB, nullable=True),
        sa.Column('training_data_start', sa.DateTime, nullable=True),
        sa.Column('training_data_end', sa.DateTime, nullable=True),
        sa.Column('training_samples', sa.Integer, nullable=True),
        sa.Column('features', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('hyperparameters', postgresql.JSONB, nullable=True),
        sa.Column('is_active', sa.Boolean, server_default='false'),
        sa.Column('deployed_at', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('created_by', sa.String(255), nullable=True),
        schema='ml'
    )
    
    # ==========================================
    # Analytics Schema Tables
    # ==========================================
    
    # recommendations table
    op.create_table(
        'recommendations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('prediction_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('action_type', sa.String(100), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('expected_risk_reduction', sa.Float, nullable=True),
        sa.Column('implementation_cost', sa.String(50), nullable=True),
        sa.Column('implementation_cost_value', sa.Float, nullable=True),
        sa.Column('roi_score', sa.Float, nullable=True),
        sa.Column('priority', sa.String(20), server_default="'MEDIUM'"),
        sa.Column('status', sa.String(20), server_default="'PENDING'"),
        sa.Column('implementation_notes', sa.Text, nullable=True),
        sa.Column('implemented_at', sa.DateTime, nullable=True),
        sa.Column('implemented_by', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['prediction_id'], ['ml.risk_predictions.id'], name='fk_recommendations_prediction'),
        schema='analytics'
    )
    
    op.create_index('idx_recommendations_status', 'recommendations', ['status'], schema='analytics')
    op.create_index('idx_recommendations_priority', 'recommendations', ['priority'], schema='analytics')
    
    # safety_kpis table
    op.create_table(
        'safety_kpis',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('kpi_type', sa.String(10), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('code', sa.String(50), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('value', sa.Float, nullable=False),
        sa.Column('unit', sa.String(50), nullable=True),
        sa.Column('measurement_date', sa.Date, nullable=False),
        sa.Column('measurement_period', sa.String(20), nullable=True),
        sa.Column('target_value', sa.Float, nullable=True),
        sa.Column('threshold_warning', sa.Float, nullable=True),
        sa.Column('threshold_critical', sa.Float, nullable=True),
        sa.Column('target_met', sa.Boolean, nullable=True),
        sa.Column('trend_direction', sa.String(20), nullable=True),
        sa.Column('sms_pillar', sa.String(30), nullable=False),
        sa.Column('category', sa.String(100), nullable=True),
        sa.Column('indicator_type', sa.String(20), server_default="'lagging'"),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        schema='analytics'
    )
    
    op.create_index('idx_kpis_date', 'safety_kpis', ['measurement_date'], schema='analytics')
    op.create_index('idx_kpis_type', 'safety_kpis', ['kpi_type'], schema='analytics')
    op.create_index('idx_kpis_pillar', 'safety_kpis', ['sms_pillar'], schema='analytics')
    
    # audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('timestamp', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('entity_type', sa.String(100), nullable=True),
        sa.Column('entity_id', sa.String(100), nullable=True),
        sa.Column('user', sa.String(255), nullable=True),
        sa.Column('details', postgresql.JSONB, nullable=True),
        sa.Column('ip_address', sa.String(50), nullable=True),
        schema='analytics'
    )
    
    op.create_index('idx_audit_timestamp', 'audit_logs', ['timestamp'], schema='analytics')
    op.create_index('idx_audit_action', 'audit_logs', ['action'], schema='analytics')
    
    # data_quality_logs table
    op.create_table(
        'data_quality_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('run_date', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('source', sa.String(50), nullable=False),
        sa.Column('total_records', sa.Integer, server_default='0'),
        sa.Column('valid_records', sa.Integer, server_default='0'),
        sa.Column('invalid_records', sa.Integer, server_default='0'),
        sa.Column('completeness_score', sa.Float, server_default='0.0'),
        sa.Column('consistency_score', sa.Float, server_default='0.0'),
        sa.Column('accuracy_score', sa.Float, server_default='0.0'),
        sa.Column('overall_score', sa.Float, server_default='0.0'),
        sa.Column('validation_errors', postgresql.JSONB, nullable=True),
        sa.Column('recommendations', postgresql.ARRAY(sa.String), nullable=True),
        schema='analytics'
    )


def downgrade() -> None:
    """Drop all tables and schemas."""
    
    # Drop analytics schema tables
    op.drop_table('data_quality_logs', schema='analytics')
    op.drop_table('audit_logs', schema='analytics')
    op.drop_table('safety_kpis', schema='analytics')
    op.drop_table('recommendations', schema='analytics')
    
    # Drop ml schema tables
    op.drop_table('model_registry', schema='ml')
    op.drop_table('risk_predictions', schema='ml')
    
    # Drop ingestion schema tables
    op.drop_table('operational_data', schema='ingestion')
    op.drop_table('weather_conditions', schema='ingestion')
    op.drop_table('incidents', schema='ingestion')
    
    # Drop schemas
    op.execute("DROP SCHEMA IF EXISTS analytics CASCADE")
    op.execute("DROP SCHEMA IF EXISTS ml CASCADE")
    op.execute("DROP SCHEMA IF EXISTS ingestion CASCADE")
