"""
AeroRisk - NTSB Data Ingestion Module

Loads aviation incident data from NTSB Microsoft Access database
into PostgreSQL for analytics.
"""

import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

import pyodbc
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.connection import engine, get_db_session
from src.database.models import Incident, DataSource, SeverityLevel, FlightPhase

load_dotenv()


class NTSBFetcher:
    """
    Fetches and processes NTSB aviation incident data from MDB files.
    
    The NTSB (National Transportation Safety Board) maintains a comprehensive
    database of aviation accidents and incidents in the United States.
    """
    
    # Mapping NTSB injury codes to our severity levels
    SEVERITY_MAPPING = {
        'FATL': SeverityLevel.FATAL,
        'SERS': SeverityLevel.SERIOUS,
        'MINR': SeverityLevel.MINOR,
        'NONE': SeverityLevel.NONE,
        None: SeverityLevel.NONE,
        '': SeverityLevel.NONE,
    }
    
    # Mapping NTSB flight phase codes
    PHASE_MAPPING = {
        'TAXI': FlightPhase.TAXI,
        'TKOF': FlightPhase.TAKEOFF,
        'CLIMB': FlightPhase.CLIMB,
        'CRUISE': FlightPhase.CRUISE,
        'DESCEND': FlightPhase.DESCENT,
        'DESCENT': FlightPhase.DESCENT,
        'APPR': FlightPhase.APPROACH,
        'APPROACH': FlightPhase.APPROACH,
        'LDG': FlightPhase.LANDING,
        'LANDING': FlightPhase.LANDING,
        'MANEUVERING': FlightPhase.CRUISE,
        'STANDING': FlightPhase.PREFLIGHT,
        'OTHER': FlightPhase.UNKNOWN,
        None: FlightPhase.UNKNOWN,
        '': FlightPhase.UNKNOWN,
    }
    
    def __init__(self, mdb_path: str):
        """
        Initialize the NTSB fetcher.
        
        Args:
            mdb_path: Path to the NTSB MDB file (avall.mdb)
        """
        self.mdb_path = Path(mdb_path)
        if not self.mdb_path.exists():
            raise FileNotFoundError(f"NTSB database not found: {mdb_path}")
        
        self.conn_str = (
            r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
            f"DBQ={self.mdb_path};"
        )
        logger.info(f"Initialized NTSBFetcher with: {self.mdb_path}")
    
    def get_connection(self):
        """Get a connection to the MDB database."""
        return pyodbc.connect(self.conn_str)
    
    def get_table_count(self, table_name: str) -> int:
        """Get row count for a table."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
            return cursor.fetchone()[0]
    
    def fetch_events(self, limit: Optional[int] = None, 
                     start_year: Optional[int] = None,
                     end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch events from the NTSB database.
        
        Args:
            limit: Maximum number of records to fetch (None for all)
            start_year: Filter events from this year onwards
            end_year: Filter events up to this year
            
        Returns:
            DataFrame with event data
        """
        query = "SELECT * FROM [events]"
        conditions = []
        
        if start_year:
            conditions.append(f"ev_year >= {start_year}")
        if end_year:
            conditions.append(f"ev_year <= {end_year}")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if limit:
            query = query.replace("SELECT *", f"SELECT TOP {limit} *")
        
        logger.info(f"Fetching events with query: {query}")
        
        with self.get_connection() as conn:
            df = pd.read_sql(query, conn)
        
        logger.info(f"Fetched {len(df)} events")
        return df
    
    def fetch_aircraft(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch aircraft details."""
        query = "SELECT * FROM [aircraft]"
        if limit:
            query = f"SELECT TOP {limit} * FROM [aircraft]"
        
        with self.get_connection() as conn:
            return pd.read_sql(query, conn)
    
    def fetch_narratives(self) -> pd.DataFrame:
        """Fetch narrative descriptions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            tables = [row.table_name for row in cursor.tables(tableType='TABLE')]
            
            # Find narratives table (case insensitive)
            narrative_table = next((t for t in tables if 'narr' in t.lower()), None)
            
            if narrative_table:
                return pd.read_sql(f"SELECT * FROM [{narrative_table}]", conn)
            else:
                logger.warning("Narratives table not found")
                return pd.DataFrame()
    
    def transform_event(self, row: pd.Series) -> Dict[str, Any]:
        """
        Transform a single NTSB event row to our Incident schema.
        
        Args:
            row: Pandas Series containing NTSB event data
            
        Returns:
            Dictionary matching our Incident model
        """
        # Parse date
        incident_date = None
        if pd.notna(row.get('ev_date')):
            incident_date = pd.to_datetime(row['ev_date'])
        
        # Parse coordinates
        latitude = None
        longitude = None
        if pd.notna(row.get('dec_latitude')):
            try:
                latitude = float(row['dec_latitude'])
            except (ValueError, TypeError):
                pass
        if pd.notna(row.get('dec_longitude')):
            try:
                longitude = float(row['dec_longitude'])
            except (ValueError, TypeError):
                pass
        
        # Build location string
        location_parts = []
        if pd.notna(row.get('ev_city')):
            location_parts.append(str(row['ev_city']))
        if pd.notna(row.get('ev_state')):
            location_parts.append(str(row['ev_state']))
        location = ', '.join(location_parts) if location_parts else None
        
        # Map severity
        injury_code = row.get('ev_highest_injury', 'NONE')
        severity = self.SEVERITY_MAPPING.get(injury_code, SeverityLevel.NONE)
        
        # Build weather data JSON
        weather_data = {}
        weather_fields = [
            ('wx_temp', 'temperature_f'),
            ('wx_dew_pt', 'dew_point_f'),
            ('wind_vel_kts', 'wind_speed_kt'),
            ('wind_dir_deg', 'wind_direction_deg'),
            ('vis_sm', 'visibility_sm'),
            ('gust_kts', 'gust_kts'),
            ('altimeter', 'altimeter'),
            ('sky_cond_ceil', 'ceiling_condition'),
            ('sky_ceil_ht', 'ceiling_height'),
            ('light_cond', 'light_condition'),
        ]
        for ntsb_field, our_field in weather_fields:
            if pd.notna(row.get(ntsb_field)):
                weather_data[our_field] = row[ntsb_field]
        
        # Parse injury counts
        def safe_int(val):
            if pd.isna(val):
                return 0
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0
        
        return {
            'source': DataSource.NTSB,
            'external_id': row.get('ev_id'),
            'incident_date': incident_date,
            'location': location,
            'airport_code': row.get('ev_nr_apt_id'),
            'country': row.get('ev_country', 'USA'),
            'latitude': latitude,
            'longitude': longitude,
            'phase_of_flight': self.PHASE_MAPPING.get(
                row.get('phase_flt_spec'), FlightPhase.UNKNOWN
            ),
            'severity': severity,
            'injuries_fatal': safe_int(row.get('inj_tot_f')),
            'injuries_serious': safe_int(row.get('inj_tot_s')),
            'injuries_minor': safe_int(row.get('inj_tot_m')),
            'injuries_uninjured': safe_int(row.get('inj_tot_n')),
            'weather_conditions': row.get('wx_cond_basic'),
            'weather_data': weather_data if weather_data else None,
            'event_type': row.get('ev_type'),
            'raw_data': {
                'ntsb_no': row.get('ntsb_no'),
                'ev_year': row.get('ev_year'),
                'ev_month': row.get('ev_month'),
                'ev_time': row.get('ev_time'),
                'ev_dow': row.get('ev_dow'),
                'apt_name': row.get('apt_name'),
            }
        }
    
    def load_to_database(self, 
                         limit: Optional[int] = None,
                         start_year: Optional[int] = None,
                         batch_size: int = 1000) -> int:
        """
        Load NTSB events into PostgreSQL database.
        
        Args:
            limit: Maximum records to load
            start_year: Only load events from this year onwards
            batch_size: Number of records per batch insert
            
        Returns:
            Number of records loaded
        """
        logger.info("Starting NTSB data load to PostgreSQL...")
        
        # Fetch events
        events_df = self.fetch_events(limit=limit, start_year=start_year)
        
        if events_df.empty:
            logger.warning("No events to load")
            return 0
        
        # Fetch aircraft for enrichment
        logger.info("Fetching aircraft data for enrichment...")
        aircraft_df = self.fetch_aircraft()
        
        # Create aircraft lookup by ev_id
        aircraft_lookup = {}
        if not aircraft_df.empty and 'ev_id' in aircraft_df.columns:
            for _, ac_row in aircraft_df.iterrows():
                ev_id = ac_row.get('ev_id')
                if ev_id and ev_id not in aircraft_lookup:
                    aircraft_lookup[ev_id] = {
                        'aircraft_make': ac_row.get('acft_make'),
                        'aircraft_model': ac_row.get('acft_model'),
                        'aircraft_type': ac_row.get('acft_category'),
                        'aircraft_registration': ac_row.get('regis_no'),
                        'operator': ac_row.get('oper_name'),
                    }
        
        logger.info(f"Aircraft lookup created with {len(aircraft_lookup)} entries")
        
        # Transform and load in batches
        loaded = 0
        errors = 0
        
        with get_db_session() as db:
            batch = []
            
            for idx, row in events_df.iterrows():
                try:
                    # Transform event
                    incident_data = self.transform_event(row)
                    
                    # Enrich with aircraft data
                    ev_id = row.get('ev_id')
                    if ev_id in aircraft_lookup:
                        incident_data.update(aircraft_lookup[ev_id])
                    
                    # Create Incident object
                    incident = Incident(**incident_data)
                    batch.append(incident)
                    
                    # Batch insert
                    if len(batch) >= batch_size:
                        db.add_all(batch)
                        db.commit()
                        loaded += len(batch)
                        logger.info(f"Loaded {loaded}/{len(events_df)} records...")
                        batch = []
                        
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        logger.error(f"Error processing event {idx}: {e}")
            
            # Final batch
            if batch:
                db.add_all(batch)
                db.commit()
                loaded += len(batch)
        
        logger.success(f"✅ Loaded {loaded} incidents to database ({errors} errors)")
        return loaded


def main():
    """Main entry point for NTSB data ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load NTSB data into AeroRisk database')
    parser.add_argument('--mdb', type=str, 
                        default='data/raw/avall.mdb',
                        help='Path to NTSB MDB file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of records to load')
    parser.add_argument('--start-year', type=int, default=None,
                        help='Only load events from this year onwards')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size for database inserts')
    
    args = parser.parse_args()
    
    # Resolve path
    mdb_path = Path(args.mdb)
    if not mdb_path.is_absolute():
        mdb_path = Path(__file__).parent.parent.parent / mdb_path
    
    print("=" * 60)
    print("🛫 AeroRisk - NTSB Data Ingestion")
    print("=" * 60)
    print(f"📁 MDB File: {mdb_path}")
    print(f"📊 Limit: {args.limit or 'All records'}")
    print(f"📅 Start Year: {args.start_year or 'All years'}")
    print()
    
    fetcher = NTSBFetcher(str(mdb_path))
    
    # Show stats
    event_count = fetcher.get_table_count('events')
    print(f"📋 Total events in database: {event_count:,}")
    print()
    
    # Load data
    loaded = fetcher.load_to_database(
        limit=args.limit,
        start_year=args.start_year,
        batch_size=args.batch_size
    )
    
    print()
    print("=" * 60)
    print(f"✅ Successfully loaded {loaded:,} incidents!")
    print("=" * 60)


if __name__ == "__main__":
    main()
