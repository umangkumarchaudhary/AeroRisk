"""Quick data verification"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

print("=" * 60)
print("📊 AeroRisk - Data Summary")
print("=" * 60)

with engine.connect() as conn:
    # Total counts
    counts = pd.read_sql(text("""
        SELECT 
            COUNT(*) as total_incidents,
            COUNT(weather_data) as with_weather,
            COUNT(latitude) as with_coords,
            COUNT(DISTINCT source) as sources
        FROM ingestion.incidents
    """), conn)
    print("\n✅ Incident Counts:")
    print(f"   Total incidents:     {counts['total_incidents'].iloc[0]:,}")
    print(f"   With weather data:   {counts['with_weather'].iloc[0]:,}")
    print(f"   With coordinates:    {counts['with_coords'].iloc[0]:,}")
    
    # By source
    source_dist = pd.read_sql(text("""
        SELECT source, COUNT(*) as count
        FROM ingestion.incidents
        GROUP BY source
        ORDER BY count DESC
    """), conn)
    print("\n📈 By Source:")
    for _, row in source_dist.iterrows():
        print(f"   {row['source']}: {row['count']:,}")
    
    # By severity
    severity_dist = pd.read_sql(text("""
        SELECT severity, COUNT(*) as count
        FROM ingestion.incidents
        GROUP BY severity
        ORDER BY count DESC
    """), conn)
    print("\n⚠️  By Severity:")
    for _, row in severity_dist.iterrows():
        print(f"   {row['severity']}: {row['count']:,}")
    
    # Weather risk by severity
    wx_risk = pd.read_sql(text("""
        SELECT 
            severity,
            AVG((weather_data->>'weather_risk_score')::numeric) as avg_risk,
            AVG((weather_data->>'visibility_m')::numeric) as avg_vis,
            AVG((weather_data->>'wind_speed_kt')::numeric) as avg_wind
        FROM ingestion.incidents
        WHERE weather_data IS NOT NULL
        GROUP BY severity
        ORDER BY avg_risk DESC
    """), conn)
    print("\n🌦️  Weather Risk by Severity:")
    print(wx_risk.round(1).to_string(index=False))
    
    # Operational data
    ops = pd.read_sql(text("""
        SELECT 
            COUNT(*) as records,
            AVG(operational_risk_score) as avg_risk
        FROM ingestion.operational_data
    """), conn)
    print(f"\n✈️  Operational Data:")
    print(f"   Records: {ops['records'].iloc[0]:,}")
    print(f"   Avg Risk Score: {ops['avg_risk'].iloc[0]:.1f}")

# Check external data files
print("\n📁 External Data Files:")
external_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'external')
if os.path.exists(external_dir):
    for f in os.listdir(external_dir):
        path = os.path.join(external_dir, f)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"   {f}: {size:,} bytes")
else:
    print("   No external data files found")

print("\n" + "=" * 60)
print("✅ Data Summary Complete!")
print("=" * 60)
