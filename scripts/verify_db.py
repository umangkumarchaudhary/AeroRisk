"""
AeroRisk - Database Verification Script

This script verifies the database setup and runs initial checks.
Run this after starting Docker containers.

Usage:
    python scripts/verify_db.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def main():
    """Run database verification."""
    
    print("=" * 60)
    print("🛫 AeroRisk - Database Verification")
    print("=" * 60)
    print()
    
    # Step 1: Test connection
    print("Step 1: Testing database connection...")
    try:
        from src.database.connection import verify_connection, get_database_info
        
        if verify_connection():
            print("   ✅ Connection successful!")
        else:
            print("   ❌ Connection failed!")
            print("\n   Make sure PostgreSQL is running:")
            print("   $ docker-compose up -d postgres")
            sys.exit(1)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("\n   Make sure you have installed dependencies:")
        print("   $ pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 2: Get database info
    print("\nStep 2: Fetching database info...")
    try:
        info = get_database_info()
        print(f"   📊 Database: {info.get('database', 'N/A')}")
        print(f"   🖥️  Host: {info.get('host', 'N/A')}:{info.get('port', 'N/A')}")
        print(f"   📁 Tables: {info.get('table_count', 'N/A')}")
        print(f"   💾 Size: {info.get('database_size', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️  Could not fetch info: {e}")
    
    # Step 3: Check schemas
    print("\nStep 3: Checking schemas...")
    try:
        from src.database.connection import engine
        from sqlalchemy import text
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name IN ('ingestion', 'ml', 'analytics')
            """))
            schemas = [row[0] for row in result.fetchall()]
            
            expected_schemas = ['ingestion', 'ml', 'analytics']
            for schema in expected_schemas:
                if schema in schemas:
                    print(f"   ✅ Schema '{schema}' exists")
                else:
                    print(f"   ⚠️  Schema '{schema}' not found - run migrations")
    except Exception as e:
        print(f"   ⚠️  Could not check schemas: {e}")
    
    # Step 4: Check tables
    print("\nStep 4: Checking core tables...")
    try:
        with engine.connect() as conn:
            # Check ingestion tables
            result = conn.execute(text("""
                SELECT table_schema, table_name 
                FROM information_schema.tables 
                WHERE table_schema IN ('ingestion', 'ml', 'analytics')
                ORDER BY table_schema, table_name
            """))
            tables = result.fetchall()
            
            if tables:
                current_schema = None
                for schema, table in tables:
                    if schema != current_schema:
                        print(f"\n   📁 {schema}/")
                        current_schema = schema
                    print(f"      └── {table}")
            else:
                print("   ⚠️  No tables found - run migrations:")
                print("   $ alembic upgrade head")
    except Exception as e:
        print(f"   ⚠️  Could not check tables: {e}")
    
    # Step 5: Test model imports
    print("\n\nStep 5: Testing model imports...")
    try:
        from src.database.models import (
            Incident, WeatherCondition, OperationalData,
            RiskPrediction, Recommendation, SafetyKPI,
            ModelRegistry, AuditLog, DataQualityLog
        )
        print("   ✅ All models imported successfully!")
        print(f"   📊 Models: Incident, WeatherCondition, OperationalData,")
        print(f"             RiskPrediction, Recommendation, SafetyKPI,")
        print(f"             ModelRegistry, AuditLog, DataQualityLog")
    except Exception as e:
        print(f"   ❌ Model import error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Database verification complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run migrations: alembic upgrade head")
    print("  2. Start Phase 2: Data Ingestion")
    print()


if __name__ == "__main__":
    main()
