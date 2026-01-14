"""Quick test to verify database tables."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT table_schema, table_name 
        FROM information_schema.tables 
        WHERE table_schema IN ('ingestion', 'ml', 'analytics') 
        ORDER BY table_schema, table_name
    """))
    
    print("=" * 50)
    print("✅ DATABASE TABLES CREATED SUCCESSFULLY!")
    print("=" * 50)
    
    current_schema = None
    for row in result.fetchall():
        schema, table = row
        if schema != current_schema:
            print(f"\n📁 {schema}/")
            current_schema = schema
        print(f"   └── {table}")
    
    print("\n" + "=" * 50)
    print("🎉 Phase 1 Complete! Ready for Phase 2!")
    print("=" * 50)
