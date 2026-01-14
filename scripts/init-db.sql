-- ============================================
-- AeroRisk Database Initialization Script
-- ============================================
-- This script runs automatically when PostgreSQL container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create schemas for better organization
CREATE SCHEMA IF NOT EXISTS ingestion;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS ml;

-- Grant permissions
GRANT ALL ON SCHEMA ingestion TO aerorisk;
GRANT ALL ON SCHEMA analytics TO aerorisk;
GRANT ALL ON SCHEMA ml TO aerorisk;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'AeroRisk database initialized successfully!';
    RAISE NOTICE 'Schemas created: ingestion, analytics, ml';
    RAISE NOTICE 'Extensions enabled: uuid-ossp, pg_trgm';
END $$;
