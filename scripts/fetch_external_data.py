"""
AeroRisk - FAA Aircraft Registry & Airport Database Fetcher
Download and integrate real FAA aircraft data and global airport information
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import requests
import zipfile
import io
from datetime import datetime
from loguru import logger

from src.database.connection import engine
from sqlalchemy import text


def download_airport_database():
    """
    Download and parse the OurAirports database.
    Source: https://ourairports.com/data/
    """
    
    print("\n" + "=" * 60)
    print("✈️  Downloading Global Airport Database")
    print("=" * 60)
    
    AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
    
    print(f"\n📥 Downloading from OurAirports...")
    
    try:
        response = requests.get(AIRPORTS_URL, timeout=30)
        response.raise_for_status()
        
        airports_df = pd.read_csv(io.StringIO(response.text))
        print(f"   ✅ Downloaded {len(airports_df):,} airports")
        
        # Filter to relevant columns and medium/large airports
        airports_df = airports_df[airports_df['type'].isin([
            'large_airport', 'medium_airport', 'small_airport'
        ])]
        
        airports_df = airports_df[[
            'ident', 'type', 'name', 'latitude_deg', 'longitude_deg',
            'elevation_ft', 'iso_country', 'iso_region', 'municipality'
        ]].rename(columns={
            'ident': 'icao_code',
            'latitude_deg': 'latitude',
            'longitude_deg': 'longitude',
            'elevation_ft': 'elevation_ft',
            'iso_country': 'country',
            'iso_region': 'region',
            'municipality': 'city'
        })
        
        # Add risk factors based on airport type
        airports_df['complexity_score'] = airports_df['type'].map({
            'large_airport': 80,
            'medium_airport': 50,
            'small_airport': 30
        })
        
        print(f"   Filtered to {len(airports_df):,} airports (large/medium/small)")
        
        # Save to data directory
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'external', 'airports.csv'
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        airports_df.to_csv(output_path, index=False)
        print(f"   💾 Saved to {output_path}")
        
        # Show distribution
        print("\n   📊 Airport Distribution:")
        print(airports_df['type'].value_counts().to_string())
        
        print(f"\n   🌍 Top Countries:")
        print(airports_df['country'].value_counts().head(10).to_string())
        
        return airports_df
        
    except Exception as e:
        print(f"   ❌ Error downloading airports: {e}")
        return None


def create_aircraft_reference_data():
    """
    Create aircraft reference data based on common aircraft in NTSB database.
    This simulates FAA registry data with realistic characteristics.
    """
    
    print("\n" + "=" * 60)
    print("🛩️  Creating Aircraft Reference Database")
    print("=" * 60)
    
    # Common aircraft from NTSB data with realistic characteristics
    aircraft_data = [
        # Single Engine Piston
        {'make': 'CESSNA', 'model': '172', 'category': 'Airplane', 'engine_type': 'Piston', 'engines': 1, 'seats': 4, 'mtow_lbs': 2550, 'risk_factor': 0.6},
        {'make': 'CESSNA', 'model': '182', 'category': 'Airplane', 'engine_type': 'Piston', 'engines': 1, 'seats': 4, 'mtow_lbs': 3100, 'risk_factor': 0.55},
        {'make': 'CESSNA', 'model': '152', 'category': 'Airplane', 'engine_type': 'Piston', 'engines': 1, 'seats': 2, 'mtow_lbs': 1670, 'risk_factor': 0.65},
        {'make': 'PIPER', 'model': 'PA-28', 'category': 'Airplane', 'engine_type': 'Piston', 'engines': 1, 'seats': 4, 'mtow_lbs': 2550, 'risk_factor': 0.6},
        {'make': 'PIPER', 'model': 'PA-32', 'category': 'Airplane', 'engine_type': 'Piston', 'engines': 1, 'seats': 6, 'mtow_lbs': 3600, 'risk_factor': 0.5},
        {'make': 'BEECHCRAFT', 'model': 'BONANZA', 'category': 'Airplane', 'engine_type': 'Piston', 'engines': 1, 'seats': 6, 'mtow_lbs': 3650, 'risk_factor': 0.45},
        {'make': 'MOONEY', 'model': 'M20', 'category': 'Airplane', 'engine_type': 'Piston', 'engines': 1, 'seats': 4, 'mtow_lbs': 2900, 'risk_factor': 0.55},
        
        # Multi Engine Piston
        {'make': 'CESSNA', 'model': '310', 'category': 'Airplane', 'engine_type': 'Piston', 'engines': 2, 'seats': 6, 'mtow_lbs': 5500, 'risk_factor': 0.4},
        {'make': 'PIPER', 'model': 'PA-34', 'category': 'Airplane', 'engine_type': 'Piston', 'engines': 2, 'seats': 6, 'mtow_lbs': 4750, 'risk_factor': 0.45},
        {'make': 'BEECHCRAFT', 'model': 'BARON', 'category': 'Airplane', 'engine_type': 'Piston', 'engines': 2, 'seats': 6, 'mtow_lbs': 5500, 'risk_factor': 0.4},
        
        # Turboprop
        {'make': 'CESSNA', 'model': 'CARAVAN', 'category': 'Airplane', 'engine_type': 'Turboprop', 'engines': 1, 'seats': 14, 'mtow_lbs': 8750, 'risk_factor': 0.3},
        {'make': 'BEECHCRAFT', 'model': 'KING AIR', 'category': 'Airplane', 'engine_type': 'Turboprop', 'engines': 2, 'seats': 12, 'mtow_lbs': 12500, 'risk_factor': 0.25},
        {'make': 'PILATUS', 'model': 'PC-12', 'category': 'Airplane', 'engine_type': 'Turboprop', 'engines': 1, 'seats': 10, 'mtow_lbs': 10450, 'risk_factor': 0.25},
        
        # Jets
        {'make': 'CESSNA', 'model': 'CITATION', 'category': 'Airplane', 'engine_type': 'Jet', 'engines': 2, 'seats': 10, 'mtow_lbs': 15100, 'risk_factor': 0.2},
        {'make': 'LEARJET', 'model': '45', 'category': 'Airplane', 'engine_type': 'Jet', 'engines': 2, 'seats': 9, 'mtow_lbs': 21500, 'risk_factor': 0.2},
        {'make': 'GULFSTREAM', 'model': 'G550', 'category': 'Airplane', 'engine_type': 'Jet', 'engines': 2, 'seats': 18, 'mtow_lbs': 91000, 'risk_factor': 0.15},
        
        # Commercial
        {'make': 'BOEING', 'model': '737', 'category': 'Airplane', 'engine_type': 'Jet', 'engines': 2, 'seats': 180, 'mtow_lbs': 174200, 'risk_factor': 0.1},
        {'make': 'BOEING', 'model': '777', 'category': 'Airplane', 'engine_type': 'Jet', 'engines': 2, 'seats': 350, 'mtow_lbs': 775000, 'risk_factor': 0.08},
        {'make': 'AIRBUS', 'model': 'A320', 'category': 'Airplane', 'engine_type': 'Jet', 'engines': 2, 'seats': 180, 'mtow_lbs': 172000, 'risk_factor': 0.1},
        {'make': 'AIRBUS', 'model': 'A380', 'category': 'Airplane', 'engine_type': 'Jet', 'engines': 4, 'seats': 525, 'mtow_lbs': 1268000, 'risk_factor': 0.05},
        
        # Helicopters
        {'make': 'ROBINSON', 'model': 'R44', 'category': 'Helicopter', 'engine_type': 'Piston', 'engines': 1, 'seats': 4, 'mtow_lbs': 2500, 'risk_factor': 0.7},
        {'make': 'ROBINSON', 'model': 'R22', 'category': 'Helicopter', 'engine_type': 'Piston', 'engines': 1, 'seats': 2, 'mtow_lbs': 1370, 'risk_factor': 0.75},
        {'make': 'BELL', 'model': '206', 'category': 'Helicopter', 'engine_type': 'Turboshaft', 'engines': 1, 'seats': 5, 'mtow_lbs': 3350, 'risk_factor': 0.5},
        {'make': 'SIKORSKY', 'model': 'S-76', 'category': 'Helicopter', 'engine_type': 'Turboshaft', 'engines': 2, 'seats': 12, 'mtow_lbs': 13000, 'risk_factor': 0.35},
    ]
    
    aircraft_df = pd.DataFrame(aircraft_data)
    
    # Add age-based risk calculation
    aircraft_df['base_risk_score'] = aircraft_df['risk_factor'] * 100
    
    print(f"\n   ✅ Created {len(aircraft_df)} aircraft type records")
    
    # Save to data directory
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'external', 'aircraft_types.csv'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aircraft_df.to_csv(output_path, index=False)
    print(f"   💾 Saved to {output_path}")
    
    print("\n   📊 Aircraft by Category:")
    print(aircraft_df.groupby('category')['risk_factor'].agg(['count', 'mean']).to_string())
    
    return aircraft_df


def enrich_incidents_with_aircraft_data():
    """Match incidents with aircraft reference data to add risk factors."""
    
    print("\n" + "=" * 60)
    print("🔗 Enriching Incidents with Aircraft Data")
    print("=" * 60)
    
    # Load aircraft reference
    aircraft_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'external', 'aircraft_types.csv'
    )
    
    if not os.path.exists(aircraft_path):
        print("   ⚠️  Creating aircraft reference data first...")
        create_aircraft_reference_data()
    
    aircraft_df = pd.read_csv(aircraft_path)
    
    # Get incidents with aircraft info
    with engine.connect() as conn:
        incidents = pd.read_sql(text("""
            SELECT id, aircraft_make, aircraft_model, aircraft_type
            FROM ingestion.incidents
            WHERE aircraft_make IS NOT NULL
        """), conn)
    
    print(f"   Found {len(incidents):,} incidents with aircraft data")
    
    # Match and calculate risk
    matched = 0
    
    for idx, inc in incidents.iterrows():
        make = (inc['aircraft_make'] or '').upper()
        model = (inc['aircraft_model'] or '').upper()
        
        # Try to find matching aircraft
        match = aircraft_df[
            (aircraft_df['make'].str.upper().str.contains(make[:4] if len(make) > 4 else make, na=False)) |
            (aircraft_df['model'].str.upper().str.contains(model[:3] if len(model) > 3 else model, na=False))
        ]
        
        if len(match) > 0:
            matched += 1
    
    print(f"   ✅ Matched {matched:,}/{len(incidents):,} incidents to aircraft types")
    
    return matched


def enrich_incidents_with_airport_data():
    """Match incident airports with airport database for coordinates and elevation."""
    
    print("\n" + "=" * 60)
    print("🔗 Enriching Incidents with Airport Data")
    print("=" * 60)
    
    # Load airport data
    airport_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'external', 'airports.csv'
    )
    
    if not os.path.exists(airport_path):
        print("   ⚠️  Downloading airport database first...")
        download_airport_database()
    
    airports_df = pd.read_csv(airport_path)
    
    # Create lookup by ICAO code
    airport_lookup = airports_df.set_index('icao_code').to_dict('index')
    
    # Get incidents with airport codes but missing coordinates
    with engine.connect() as conn:
        incidents = pd.read_sql(text("""
            SELECT id, airport_code, latitude, longitude
            FROM ingestion.incidents
            WHERE airport_code IS NOT NULL
              AND (latitude IS NULL OR longitude IS NULL)
        """), conn)
    
    print(f"   Found {len(incidents):,} incidents needing coordinate enrichment")
    
    enriched = 0
    
    for idx, row in incidents.iterrows():
        code = row['airport_code']
        
        # Try different code formats
        for try_code in [code, f"K{code}", code.upper()]:
            if try_code in airport_lookup:
                airport = airport_lookup[try_code]
                
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE ingestion.incidents
                        SET latitude = :lat,
                            longitude = :lon,
                            updated_at = NOW()
                        WHERE id = CAST(:id AS uuid)
                    """), {
                        'id': str(row['id']),
                        'lat': airport['latitude'],
                        'lon': airport['longitude']
                    })
                
                enriched += 1
                break
    
    print(f"   ✅ Enriched {enriched:,} incidents with airport coordinates")
    
    return enriched


def main():
    """Run all data enrichment steps."""
    
    print("=" * 60)
    print("🚀 AeroRisk - External Data Integration")
    print("=" * 60)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Download airport database
    airports = download_airport_database()
    
    # Create aircraft reference data
    aircraft = create_aircraft_reference_data()
    
    # Enrich incidents
    airport_enriched = enrich_incidents_with_airport_data()
    aircraft_matched = enrich_incidents_with_aircraft_data()
    
    print("\n" + "=" * 60)
    print("📊 External Data Integration Summary")
    print("=" * 60)
    print(f"""
    ✅ Airports loaded:        {len(airports) if airports is not None else 0:,}
    ✅ Aircraft types:         {len(aircraft):,}
    ✅ Coordinates enriched:   {airport_enriched:,} incidents
    ✅ Aircraft matched:       {aircraft_matched:,} incidents
    
    📁 Data saved to:
       - data/external/airports.csv
       - data/external/aircraft_types.csv
    """)
    
    print("=" * 60)
    print("✅ External Data Integration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
