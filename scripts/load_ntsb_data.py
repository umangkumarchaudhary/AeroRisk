"""
Load NTSB Data - Using pandas to_sql for reliable bulk insert
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pyodbc
from datetime import datetime
import uuid
from loguru import logger

from src.database.connection import engine
from sqlalchemy import text

# Path to MDB
mdb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "data", "raw", "avall.mdb")

print("=" * 60)
print("🛫 AeroRisk - NTSB Data Load (Direct SQL)")
print("=" * 60)
print(f"📁 Source: {mdb_path}")

# Connect to Access database
conn_str = (
    r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
    f"DBQ={mdb_path};"
)

print("\n📊 Reading NTSB events...")
mdb_conn = pyodbc.connect(conn_str)
events_df = pd.read_sql("SELECT * FROM [events]", mdb_conn)
print(f"   Loaded {len(events_df):,} events from MDB")

print("\n✈️  Reading aircraft data...")
aircraft_df = pd.read_sql("SELECT * FROM [aircraft]", mdb_conn)
print(f"   Loaded {len(aircraft_df):,} aircraft records")
mdb_conn.close()

# Create aircraft lookup
aircraft_lookup = {}
for _, row in aircraft_df.iterrows():
    ev_id = row.get('ev_id')
    if ev_id and ev_id not in aircraft_lookup:
        aircraft_lookup[ev_id] = {
            'aircraft_make': row.get('acft_make'),
            'aircraft_model': row.get('acft_model'),
            'aircraft_type': row.get('acft_category'),
            'aircraft_registration': row.get('regis_no'),
            'operator': row.get('oper_name'),
        }

print(f"   Created lookup for {len(aircraft_lookup):,} aircraft")

# Severity mapping
SEVERITY_MAP = {
    'FATL': 'FATAL',
    'SERS': 'SERIOUS', 
    'MINR': 'MINOR',
    'NONE': 'NONE',
}

def safe_int(val):
    if pd.isna(val):
        return 0
    try:
        return int(val)
    except:
        return 0

def safe_float(val):
    if pd.isna(val):
        return None
    try:
        return float(val)
    except:
        return None

def safe_str(val):
    if pd.isna(val):
        return None
    return str(val)[:255] if val else None

print("\n🔄 Transforming data...")

# Transform to our schema
incidents = []
for idx, row in events_df.iterrows():
    try:
        # Build location
        location_parts = []
        if pd.notna(row.get('ev_city')):
            location_parts.append(str(row['ev_city']))
        if pd.notna(row.get('ev_state')):
            location_parts.append(str(row['ev_state']))
        location = ', '.join(location_parts) if location_parts else None
        
        # Get aircraft info
        ev_id = row.get('ev_id')
        ac_info = aircraft_lookup.get(ev_id, {})
        
        # Map severity
        injury_code = row.get('ev_highest_injury', 'NONE')
        severity = SEVERITY_MAP.get(injury_code, 'NONE')
        
        incident = {
            'id': str(uuid.uuid4()),
            'source': 'NTSB',
            'external_id': safe_str(row.get('ev_id')),
            'incident_date': row.get('ev_date'),
            'report_date': None,
            'location': location,
            'airport_code': safe_str(row.get('ev_nr_apt_id')),
            'country': safe_str(row.get('ev_country')) or 'USA',
            'latitude': safe_float(row.get('dec_latitude')),
            'longitude': safe_float(row.get('dec_longitude')),
            'aircraft_type': safe_str(ac_info.get('aircraft_type')),
            'aircraft_make': safe_str(ac_info.get('aircraft_make')),
            'aircraft_model': safe_str(ac_info.get('aircraft_model')),
            'aircraft_registration': safe_str(ac_info.get('aircraft_registration')),
            'operator': safe_str(ac_info.get('operator')),
            'phase_of_flight': 'UNKNOWN',
            'flight_type': safe_str(row.get('type_fly')),
            'severity': severity,
            'injuries_fatal': safe_int(row.get('inj_tot_f')),
            'injuries_serious': safe_int(row.get('inj_tot_s')),
            'injuries_minor': safe_int(row.get('inj_tot_m')),
            'injuries_uninjured': safe_int(row.get('inj_tot_n')),
            'aircraft_damage': safe_str(row.get('damage')),
            'probable_cause': None,
            'contributing_factors': None,
            'event_type': safe_str(row.get('ev_type')),
            'weather_conditions': safe_str(row.get('wx_cond_basic')),
            'weather_data': None,
            'raw_data': None,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
        }
        incidents.append(incident)
        
        if len(incidents) % 5000 == 0:
            print(f"   Processed {len(incidents):,}/{len(events_df):,} events...")
            
    except Exception as e:
        if idx < 5:
            print(f"   Error at row {idx}: {e}")

print(f"   ✅ Transformed {len(incidents):,} incidents")

# Create DataFrame
incidents_df = pd.DataFrame(incidents)

print("\n💾 Loading to PostgreSQL...")

# Insert in chunks using pandas to_sql
chunk_size = 5000
total_loaded = 0

for i in range(0, len(incidents_df), chunk_size):
    chunk = incidents_df.iloc[i:i+chunk_size]
    chunk.to_sql(
        'incidents',
        engine,
        schema='ingestion',
        if_exists='append',
        index=False,
        method='multi'
    )
    total_loaded += len(chunk)
    print(f"   Loaded {total_loaded:,}/{len(incidents_df):,} records...")

print(f"\n✅ Successfully loaded {total_loaded:,} incidents!")

# Verify
print("\n📊 Verifying in database...")
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM ingestion.incidents"))
    count = result.fetchone()[0]
    print(f"   Total incidents in PostgreSQL: {count:,}")
    
    result = conn.execute(text("""
        SELECT severity, COUNT(*) as cnt 
        FROM ingestion.incidents 
        GROUP BY severity 
        ORDER BY cnt DESC
    """))
    print("\n   📈 Severity Distribution:")
    for row in result.fetchall():
        print(f"      {row[0]}: {row[1]:,}")
        
    result = conn.execute(text("""
        SELECT EXTRACT(YEAR FROM incident_date) as year, COUNT(*) as cnt
        FROM ingestion.incidents
        WHERE incident_date IS NOT NULL
        GROUP BY EXTRACT(YEAR FROM incident_date)
        ORDER BY year DESC
        LIMIT 10
    """))
    print("\n   📅 Recent Years:")
    for row in result.fetchall():
        print(f"      {int(row[0])}: {row[1]:,} incidents")

print("\n" + "=" * 60)
print("🎉 NTSB Data Load Complete!")
print("=" * 60)
