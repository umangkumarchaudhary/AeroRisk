"""
AeroRisk - Synthetic Operational Data Generator

Generates realistic operational data (crew duty, maintenance, schedules)
for training predictive models. This simulates the confidential operational
data that airlines maintain internally.
"""

import os
import sys
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.connection import engine
from sqlalchemy import text


class SyntheticDataGenerator:
    """
    Generates realistic synthetic operational data for aviation analytics.
    
    Data is generated based on:
    - FAA duty time regulations
    - Industry maintenance standards
    - Realistic flight scheduling patterns
    """
    
    # Major US airports for route generation
    AIRPORTS = [
        'LAX', 'JFK', 'ORD', 'DFW', 'DEN', 'SFO', 'SEA', 'LAS', 'MIA', 'ATL',
        'BOS', 'PHX', 'IAH', 'MSP', 'DTW', 'EWR', 'MCO', 'CLT', 'PHL', 'SAN',
        'SLC', 'DCA', 'IAD', 'BWI', 'TPA', 'PDX', 'STL', 'HNL', 'AUS', 'RDU'
    ]
    
    # Aircraft types with age distribution
    AIRCRAFT_TYPES = [
        {'type': 'B737', 'make': 'Boeing', 'model': '737-800', 'avg_age': 12},
        {'type': 'B777', 'make': 'Boeing', 'model': '777-300ER', 'avg_age': 10},
        {'type': 'B787', 'make': 'Boeing', 'model': '787-9', 'avg_age': 5},
        {'type': 'A320', 'make': 'Airbus', 'model': 'A320neo', 'avg_age': 4},
        {'type': 'A321', 'make': 'Airbus', 'model': 'A321neo', 'avg_age': 3},
        {'type': 'A350', 'make': 'Airbus', 'model': 'A350-900', 'avg_age': 4},
        {'type': 'E175', 'make': 'Embraer', 'model': 'E175', 'avg_age': 8},
        {'type': 'CRJ9', 'make': 'Bombardier', 'model': 'CRJ-900', 'avg_age': 10},
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        logger.info(f"Initialized SyntheticDataGenerator with seed={seed}")
    
    def generate_aircraft_id(self) -> str:
        """Generate realistic aircraft registration number."""
        # US registrations start with N
        return f"N{random.randint(100, 999)}{random.choice(['UA', 'DL', 'AA', 'SW', 'UN'])}"
    
    def generate_flight_id(self, date: datetime) -> str:
        """Generate realistic flight number."""
        prefix = random.choice(['UA', 'DL', 'AA', 'WN', 'B6', 'AS'])
        number = random.randint(100, 9999)
        return f"{prefix}{number}"
    
    def generate_crew_data(self) -> Dict[str, Any]:
        """
        Generate crew duty and fatigue data.
        
        Based on FAA regulations:
        - Max flight duty period: 9-14 hours depending on start time
        - Required rest: minimum 10 hours between duty periods
        - Monthly limits: 100 flight hours, 1000 duty hours
        """
        # Duty hours (0-14 with realistic distribution)
        duty_hours = np.random.gamma(4, 2)  # Skewed towards 8-10 hours
        duty_hours = min(max(duty_hours, 0), 14)
        
        # Rest hours before duty
        rest_hours = np.random.gamma(3, 3)  # Should be 10+ for legal
        rest_hours = min(max(rest_hours, 6), 24)
        
        # Fatigue index (0-100, calculated based on factors)
        # Higher = more fatigued
        fatigue_base = 20  # Baseline
        fatigue_duty = (duty_hours / 14) * 30  # Duty contribution
        fatigue_rest = max(0, (10 - rest_hours)) * 5  # Insufficient rest
        fatigue_random = np.random.normal(0, 10)  # Individual variation
        
        fatigue_index = fatigue_base + fatigue_duty + fatigue_rest + fatigue_random
        fatigue_index = min(max(fatigue_index, 0), 100)
        
        # Pilot experience (hours)
        experience_hours = np.random.lognormal(8, 1)  # Log-normal: most pilots 1000-5000 hrs
        experience_hours = min(experience_hours, 25000)
        
        return {
            'crew_duty_hours': round(duty_hours, 1),
            'crew_rest_hours': round(rest_hours, 1),
            'crew_fatigue_index': round(fatigue_index, 1),
            'pilot_experience_hours': int(experience_hours),
        }
    
    def generate_maintenance_data(self, aircraft_age: float) -> Dict[str, Any]:
        """
        Generate maintenance status data.
        
        Older aircraft need more maintenance and have higher
        probability of overdue items.
        """
        # Days since major maintenance (A/B/C checks)
        # C-check every 18-24 months
        days_since_major = int(np.random.exponential(200))
        
        # Probability of overdue increases with aircraft age
        overdue_prob = 0.02 + (aircraft_age / 100) * 0.08
        maintenance_overdue = random.random() < overdue_prob
        
        # Open maintenance items (deferred items, MELs)
        open_items_lambda = 0.5 + aircraft_age / 20
        open_items = int(np.random.poisson(open_items_lambda))
        
        return {
            'aircraft_age_years': round(aircraft_age, 1),
            'days_since_major_maintenance': days_since_major,
            'maintenance_overdue_flag': maintenance_overdue,
            'open_maintenance_items': open_items,
        }
    
    def generate_schedule_data(self, scheduled_time: datetime) -> Dict[str, Any]:
        """
        Generate schedule-related data.
        
        Includes delays, turnaround times, etc.
        """
        # Schedule deviation (negative = early, positive = late)
        # Most flights on time or slightly delayed
        deviation = int(np.random.normal(5, 15))  # Mean 5 min late
        deviation = max(-30, min(deviation, 180))  # Cap at -30 to +180 mins
        
        actual_time = scheduled_time + timedelta(minutes=deviation)
        
        # Turnaround time (time between landing and next departure)
        turnaround = int(np.random.gamma(6, 10))  # Mean ~60 mins
        turnaround = max(30, min(turnaround, 180))
        
        return {
            'scheduled_departure': scheduled_time,
            'actual_departure': actual_time,
            'schedule_deviation_minutes': deviation,
            'turnaround_time_minutes': turnaround,
        }
    
    def calculate_operational_risk(self, crew: Dict, maintenance: Dict, 
                                   schedule: Dict) -> float:
        """
        Calculate overall operational risk score based on all factors.
        
        Score 0-100 where higher = more risk.
        """
        risk = 0
        
        # Crew fatigue contribution (30% weight)
        risk += (crew['crew_fatigue_index'] / 100) * 30
        
        # Maintenance contribution (40% weight)
        if maintenance['maintenance_overdue_flag']:
            risk += 20
        risk += (maintenance['open_maintenance_items'] / 10) * 10
        risk += min((maintenance['aircraft_age_years'] / 30) * 10, 10)
        
        # Schedule contribution (30% weight)
        delay = abs(schedule['schedule_deviation_minutes'])
        risk += min((delay / 60) * 15, 15)
        if schedule['turnaround_time_minutes'] < 45:
            risk += 10  # Short turnaround = more risk
        
        # Add some randomness
        risk += np.random.normal(0, 5)
        
        return round(min(max(risk, 0), 100), 1)
    
    def generate_records(self, num_records: int = 50000,
                         start_date: datetime = None,
                         end_date: datetime = None) -> pd.DataFrame:
        """
        Generate synthetic operational data records.
        
        Args:
            num_records: Number of records to generate
            start_date: Start of date range (default: 2020-01-01)
            end_date: End of date range (default: today)
            
        Returns:
            DataFrame with operational data
        """
        if start_date is None:
            start_date = datetime(2020, 1, 1)
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Generating {num_records:,} synthetic operational records...")
        
        records = []
        date_range = (end_date - start_date).days
        
        # Pre-generate aircraft fleet
        fleet_size = 200
        fleet = []
        for i in range(fleet_size):
            ac_type = random.choice(self.AIRCRAFT_TYPES)
            age = max(0, np.random.normal(ac_type['avg_age'], 3))
            fleet.append({
                'aircraft_id': self.generate_aircraft_id(),
                'type': ac_type['type'],
                'make': ac_type['make'],
                'model': ac_type['model'],
                'age': age,
            })
        
        for i in range(num_records):
            # Random date within range
            days_offset = random.randint(0, date_range)
            record_date = start_date + timedelta(days=days_offset)
            
            # Random time of day (flights more common during day)
            hour = int(np.random.normal(12, 4))
            hour = max(0, min(hour, 23))
            scheduled_time = record_date.replace(hour=hour, minute=random.randint(0, 59))
            
            # Random aircraft
            aircraft = random.choice(fleet)
            
            # Random route
            origin = random.choice(self.AIRPORTS)
            destination = random.choice([a for a in self.AIRPORTS if a != origin])
            
            # Generate component data
            crew_data = self.generate_crew_data()
            maintenance_data = self.generate_maintenance_data(aircraft['age'])
            schedule_data = self.generate_schedule_data(scheduled_time)
            
            # Calculate risk
            risk_score = self.calculate_operational_risk(
                crew_data, maintenance_data, schedule_data
            )
            
            record = {
                'id': str(uuid.uuid4()),
                'date': record_date.date(),
                'flight_id': self.generate_flight_id(record_date),
                'aircraft_id': aircraft['aircraft_id'],
                'origin_airport': origin,
                'destination_airport': destination,
                **crew_data,
                **maintenance_data,
                **schedule_data,
                'operational_risk_score': risk_score,
                'created_at': datetime.now(),
            }
            records.append(record)
            
            if (i + 1) % 10000 == 0:
                logger.info(f"Generated {i + 1:,}/{num_records:,} records...")
        
        df = pd.DataFrame(records)
        logger.success(f"✅ Generated {len(df):,} synthetic operational records")
        return df
    
    def load_to_database(self, df: pd.DataFrame) -> int:
        """Load generated data to PostgreSQL."""
        logger.info("Loading synthetic data to PostgreSQL...")
        
        # Insert in chunks
        chunk_size = 5000
        total = 0
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk.to_sql(
                'operational_data',
                engine,
                schema='ingestion',
                if_exists='append',
                index=False,
                method='multi'
            )
            total += len(chunk)
            logger.info(f"Loaded {total:,}/{len(df):,} records...")
        
        logger.success(f"✅ Loaded {total:,} records to database")
        return total


def main():
    """Generate and load synthetic operational data."""
    print("=" * 60)
    print("🏭 AeroRisk - Synthetic Data Generator")
    print("=" * 60)
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate 50K records (2020-2025)
    df = generator.generate_records(
        num_records=50000,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2025, 12, 31)
    )
    
    print(f"\n📊 Generated data summary:")
    print(f"   Records: {len(df):,}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Unique aircraft: {df['aircraft_id'].nunique()}")
    print(f"   Unique flights: {df['flight_id'].nunique()}")
    print(f"\n   Risk score distribution:")
    print(f"      Min: {df['operational_risk_score'].min():.1f}")
    print(f"      Max: {df['operational_risk_score'].max():.1f}")
    print(f"      Mean: {df['operational_risk_score'].mean():.1f}")
    print(f"      High risk (>70): {(df['operational_risk_score'] > 70).sum():,}")
    
    # Load to database
    print("\n💾 Loading to PostgreSQL...")
    loaded = generator.load_to_database(df)
    
    # Verify
    print("\n📊 Verifying...")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM ingestion.operational_data"))
        count = result.fetchone()[0]
        print(f"   Total in database: {count:,}")
    
    print("\n" + "=" * 60)
    print("🎉 Synthetic data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
