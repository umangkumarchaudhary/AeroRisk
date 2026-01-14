"""
Load ASRS (Aviation Safety Reporting System) Data from TSV
Near-miss reports that complement NTSB accident/incident data

Note: The .xls file is actually a TSV (Tab-Separated Values) text file
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
import uuid
from loguru import logger

from src.database.connection import engine
from sqlalchemy import text

# Path to ASRS file (TSV format with .xls extension)
asrs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "data", "raw", "ASRS_DBOnline.xls")

print("=" * 60)
print("🛫 AeroRisk - ASRS Near-Miss Data Load")
print("=" * 60)
print(f"📁 Source: {asrs_path}")

print("\n📊 Reading ASRS data as TSV...")
try:
    # Read as TSV - the file has merged header rows
    # Skip first row (merged category headers), use second row as column names
    df = pd.read_csv(asrs_path, sep='\t', header=0, encoding='utf-8', on_bad_lines='skip')
    print(f"   ✅ Loaded {len(df):,} ASRS reports")
    print(f"   Columns found: {len(df.columns)}")
except Exception as e:
    print(f"   Error with UTF-8: {e}")
    print("   Trying latin-1 encoding...")
    df = pd.read_csv(asrs_path, sep='\t', header=0, encoding='latin-1', on_bad_lines='skip')
    print(f"   ✅ Loaded {len(df):,} ASRS reports")

# Display first few columns to understand structure
print("\n📋 First 30 columns:")
for i, col in enumerate(df.columns[:30]):
    print(f"   {i}: '{col}'")

print("\n📋 Last 20 columns:")
for i, col in enumerate(df.columns[-20:]):
    print(f"   {len(df.columns)-20+i}: '{col}'")

# Based on the column structure provided:
# Time (Date, Local Time Of Day)
# Place (State Reference, Altitude, etc.)
# Environment (Flight Conditions, Weather, Light)
# Aircraft 1 (Operator, Make Model, Flight Phase, etc.)
# Events (Anomaly, Result)
# Assessments (Primary Problem)
# Report (Narrative, Synopsis)

# Severity mapping for near-miss (most are near-miss, no injuries)
def map_severity(result, anomaly):
    """Map ASRS event type to severity - near misses are typically minor"""
    result_str = str(result).lower() if pd.notna(result) else ''
    anomaly_str = str(anomaly).lower() if pd.notna(anomaly) else ''
    
    if 'collision' in result_str or 'collision' in anomaly_str:
        return 'SERIOUS'
    elif 'deviation' in result_str or 'excursion' in result_str:
        return 'MINOR'
    else:
        return 'NONE'  # Near-miss without actual damage

def map_flight_phase(phase_str):
    """Map ASRS flight phase to our enum"""
    if pd.isna(phase_str):
        return 'UNKNOWN'
    
    phase_lower = str(phase_str).lower()
    phase_map = {
        'takeoff': 'TAKEOFF',
        'climb': 'CLIMB',
        'cruise': 'CRUISE',
        'descent': 'DESCENT',
        'approach': 'APPROACH',
        'landing': 'LANDING',
        'taxi': 'TAXI',
        'parked': 'PREFLIGHT',
        'initial': 'CLIMB',
        'final': 'APPROACH',
    }
    
    for key, value in phase_map.items():
        if key in phase_lower:
            return value
    return 'UNKNOWN'

def safe_str(val, max_len=255):
    if pd.isna(val):
        return None
    return str(val)[:max_len] if val else None

def safe_float(val):
    if pd.isna(val):
        return None
    try:
        return float(val)
    except:
        return None

def parse_date(date_val):
    """Parse various date formats from ASRS data"""
    if pd.isna(date_val):
        return None
    
    if isinstance(date_val, datetime):
        return date_val
    
    try:
        date_str = str(date_val).strip()
        # Try common formats
        for fmt in ['%Y%m', '%Y-%m', '%m/%Y', '%Y', '%Y%m%d', '%m/%d/%Y']:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        # Handle YYYYMM format (e.g., 202301)
        if date_str.isdigit():
            if len(date_str) == 6:  # YYYYMM
                return datetime(int(date_str[:4]), int(date_str[4:6]), 1)
            elif len(date_str) == 4:  # YYYY
                return datetime(int(date_str), 1, 1)
        return None
    except:
        return None

print("\n🔄 Transforming ASRS data...")

# Build column name mapping - find columns by partial match
def find_column(df, patterns, exact=False):
    """Find column matching any of the patterns"""
    for col in df.columns:
        col_str = str(col)
        for pattern in patterns:
            if exact:
                if col_str.strip() == pattern:
                    return col
            else:
                if pattern.lower() in col_str.lower():
                    return col
    return None

# Identify key columns based on provided structure
col_date = find_column(df, ['Date'])
col_state = find_column(df, ['State Reference', 'State'])
col_altitude = find_column(df, ['Altitude.MSL', 'Altitude'])
col_conditions = find_column(df, ['Flight Conditions'])
col_weather = find_column(df, ['Weather Elements'])
col_light = find_column(df, ['Light'])
col_operator = find_column(df, ['Aircraft Operator'])
col_make_model = find_column(df, ['Make Model Name'])
col_phase = find_column(df, ['Flight Phase'])
col_airspace = find_column(df, ['Airspace'])
col_human_factors = find_column(df, ['Human Factors'])
col_anomaly = find_column(df, ['Anomaly'])
col_result = find_column(df, ['Result'])
col_problem = find_column(df, ['Primary Problem'])
col_narrative = find_column(df, ['Narrative'])
col_synopsis = find_column(df, ['Synopsis'])
col_acn = find_column(df, ['ACN', 'Accession Number', 'Report Number'])

print("\n📋 Column mapping found:")
print(f"   Date: {col_date}")
print(f"   State: {col_state}")
print(f"   Aircraft: {col_make_model}")
print(f"   Operator: {col_operator}")
print(f"   Phase: {col_phase}")
print(f"   Anomaly: {col_anomaly}")
print(f"   Result: {col_result}")
print(f"   Synopsis: {col_synopsis}")
print(f"   ACN: {col_acn}")

# Transform to incidents
incidents = []
for idx, row in df.iterrows():
    try:
        # Parse date
        incident_date = parse_date(row.get(col_date)) if col_date else None
        if incident_date is None:
            incident_date = datetime(2020, 1, 1)  # Default date for records without date
        
        # Get narrative/synopsis for probable cause
        narrative = safe_str(row.get(col_narrative), 5000) if col_narrative else None
        synopsis = safe_str(row.get(col_synopsis), 2000) if col_synopsis else None
        probable_cause = synopsis or narrative
        
        # Get anomaly and result for severity
        anomaly = row.get(col_anomaly) if col_anomaly else None
        result = row.get(col_result) if col_result else None
        severity = map_severity(result, anomaly)
        
        # Build location
        state = safe_str(row.get(col_state)) if col_state else None
        location = state if state else 'USA'
        
        # Get external ID
        acn = row.get(col_acn) if col_acn else None
        external_id = safe_str(acn) if acn else f"ASRS-{idx}"
        
        incident = {
            'id': str(uuid.uuid4()),
            'source': 'ASRS',
            'external_id': external_id,
            'incident_date': incident_date,
            'report_date': None,
            'location': location,
            'airport_code': None,
            'country': 'USA',
            'latitude': None,
            'longitude': None,
            'aircraft_type': None,
            'aircraft_make': safe_str(row.get(col_make_model)) if col_make_model else None,
            'aircraft_model': None,
            'aircraft_registration': None,
            'operator': safe_str(row.get(col_operator)) if col_operator else None,
            'phase_of_flight': map_flight_phase(row.get(col_phase)) if col_phase else 'UNKNOWN',
            'flight_type': None,
            'severity': severity,
            'injuries_fatal': 0,  # Near-miss reports typically have no injuries
            'injuries_serious': 0,
            'injuries_minor': 0,
            'injuries_uninjured': 0,
            'aircraft_damage': 'None',  # Near-miss
            'probable_cause': probable_cause,
            'contributing_factors': None,
            'event_type': safe_str(anomaly)[:100] if anomaly else 'Near Miss',
            'weather_conditions': safe_str(row.get(col_conditions)) if col_conditions else None,
            'weather_data': None,
            'raw_data': None,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
        }
        incidents.append(incident)
        
        if len(incidents) % 5000 == 0:
            print(f"   Processed {len(incidents):,}/{len(df):,} reports...")
            
    except Exception as e:
        if idx < 5:
            print(f"   Error at row {idx}: {e}")

print(f"   ✅ Transformed {len(incidents):,} ASRS reports")

if len(incidents) == 0:
    print("\n❌ No incidents to load! Please check the data format.")
    sys.exit(1)

# Create DataFrame
incidents_df = pd.DataFrame(incidents)

print("\n💾 Loading to PostgreSQL...")

# Insert in chunks
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

print(f"\n✅ Successfully loaded {total_loaded:,} ASRS reports!")

# Verify
print("\n📊 Verifying in database...")
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM ingestion.incidents WHERE source = 'ASRS'"))
    count = result.fetchone()[0]
    print(f"   ASRS incidents in PostgreSQL: {count:,}")
    
    result = conn.execute(text("SELECT COUNT(*) FROM ingestion.incidents"))
    total = result.fetchone()[0]
    print(f"   Total incidents (all sources): {total:,}")
    
    result = conn.execute(text("""
        SELECT source, COUNT(*) as cnt 
        FROM ingestion.incidents 
        GROUP BY source 
        ORDER BY cnt DESC
    """))
    print("\n   📈 Incidents by Source:")
    for row in result.fetchall():
        print(f"      {row[0]}: {row[1]:,}")

print("\n" + "=" * 60)
print("🎉 ASRS Data Load Complete!")
print("=" * 60)
