"""
AeroRisk - Synthetic Weather Data Generator
Generate realistic weather data correlated with incident patterns
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from loguru import logger

from src.database.connection import engine
from sqlalchemy import text


class WeatherGenerator:
    """Generate realistic synthetic weather data based on incident characteristics."""
    
    # Weather condition distributions based on aviation statistics
    # Higher severity incidents correlate with worse weather
    WEATHER_CONDITIONS = {
        'FATAL': {
            'conditions': ['Thunderstorm', 'Fog', 'Snow', 'Rain', 'Clouds', 'Clear'],
            'weights': [0.15, 0.20, 0.15, 0.20, 0.20, 0.10],
            'visibility_range': (500, 8000),
            'wind_range': (10, 35),
        },
        'SERIOUS': {
            'conditions': ['Thunderstorm', 'Fog', 'Snow', 'Rain', 'Clouds', 'Clear'],
            'weights': [0.10, 0.15, 0.12, 0.25, 0.25, 0.13],
            'visibility_range': (1000, 10000),
            'wind_range': (8, 30),
        },
        'MINOR': {
            'conditions': ['Thunderstorm', 'Fog', 'Snow', 'Rain', 'Clouds', 'Clear'],
            'weights': [0.05, 0.08, 0.08, 0.20, 0.35, 0.24],
            'visibility_range': (3000, 15000),
            'wind_range': (5, 25),
        },
        'NONE': {
            'conditions': ['Thunderstorm', 'Fog', 'Snow', 'Rain', 'Clouds', 'Clear'],
            'weights': [0.02, 0.05, 0.05, 0.15, 0.40, 0.33],
            'visibility_range': (5000, 20000),
            'wind_range': (0, 20),
        }
    }
    
    # Seasonal temperature ranges (Northern Hemisphere averages)
    SEASONAL_TEMPS = {
        1: (-5, 10),   # January
        2: (-3, 12),   # February
        3: (2, 18),    # March
        4: (8, 22),    # April
        5: (14, 28),   # May
        6: (18, 32),   # June
        7: (20, 35),   # July
        8: (19, 34),   # August
        9: (14, 28),   # September
        10: (8, 22),   # October
        11: (2, 15),   # November
        12: (-3, 10),  # December
    }
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def generate_for_incident(
        self, 
        severity: str, 
        incident_date: datetime,
        latitude: float = None,
        longitude: float = None
    ) -> dict:
        """Generate weather data for a single incident."""
        
        severity = severity if severity in self.WEATHER_CONDITIONS else 'NONE'
        config = self.WEATHER_CONDITIONS[severity]
        
        # Generate weather condition
        condition = np.random.choice(config['conditions'], p=config['weights'])
        
        # Generate visibility
        vis_min, vis_max = config['visibility_range']
        visibility = np.random.uniform(vis_min, vis_max)
        
        # Adjust visibility based on condition
        if condition == 'Fog':
            visibility = min(visibility, 1000)
        elif condition == 'Thunderstorm':
            visibility = min(visibility, 5000)
        elif condition == 'Clear':
            visibility = max(visibility, 10000)
        
        # Generate wind
        wind_min, wind_max = config['wind_range']
        wind_speed = np.random.uniform(wind_min, wind_max)
        wind_gust = wind_speed + np.random.uniform(0, 10) if wind_speed > 15 else None
        wind_direction = np.random.randint(0, 360)
        
        # Generate temperature based on month
        month = incident_date.month if incident_date else 6
        temp_min, temp_max = self.SEASONAL_TEMPS[month]
        temperature = np.random.uniform(temp_min, temp_max)
        
        # Adjust for latitude if available (colder further from equator)
        if latitude:
            lat_factor = (abs(latitude) - 40) / 50  # Deviation from temperate zone
            temperature -= lat_factor * 5
        
        # Generate other parameters
        humidity = np.random.uniform(40, 95)
        if condition in ['Rain', 'Fog', 'Snow']:
            humidity = np.random.uniform(70, 100)
        
        pressure = np.random.uniform(1000, 1030)
        if condition == 'Thunderstorm':
            pressure = np.random.uniform(980, 1010)
        
        cloud_cover = {
            'Clear': np.random.uniform(0, 20),
            'Clouds': np.random.uniform(50, 100),
            'Rain': np.random.uniform(70, 100),
            'Snow': np.random.uniform(80, 100),
            'Fog': np.random.uniform(90, 100),
            'Thunderstorm': np.random.uniform(80, 100),
        }.get(condition, 50)
        
        # Calculate weather risk score
        weather_risk = self._calculate_risk(
            visibility, wind_speed, condition, temperature
        )
        
        # Determine VMC/IMC
        is_imc = visibility < 5000 or cloud_cover > 80
        
        return {
            'temperature_c': round(temperature, 1),
            'feels_like_c': round(temperature - (wind_speed * 0.3), 1),
            'humidity_percent': round(humidity, 1),
            'pressure_hpa': round(pressure, 1),
            'visibility_m': round(visibility, 0),
            'cloud_cover_percent': round(cloud_cover, 1),
            'wind_speed_kt': round(wind_speed, 1),
            'wind_gust_kt': round(wind_gust, 1) if wind_gust else None,
            'wind_direction_deg': wind_direction,
            'weather_condition': condition,
            'severe_weather_flag': condition in ['Thunderstorm', 'Tornado'] or wind_speed > 25,
            'weather_risk_score': round(weather_risk, 1),
            'flight_conditions': 'IMC' if is_imc else 'VMC',
            'source': 'SYNTHETIC'
        }
    
    def _calculate_risk(
        self, 
        visibility: float, 
        wind_speed: float, 
        condition: str,
        temperature: float
    ) -> float:
        """Calculate weather risk score 0-100."""
        risk = 0.0
        
        # Visibility risk (0-30)
        if visibility < 500:
            risk += 30
        elif visibility < 1000:
            risk += 25
        elif visibility < 3000:
            risk += 18
        elif visibility < 5000:
            risk += 12
        elif visibility < 8000:
            risk += 6
        
        # Wind risk (0-30)
        if wind_speed > 30:
            risk += 30
        elif wind_speed > 25:
            risk += 25
        elif wind_speed > 20:
            risk += 18
        elif wind_speed > 15:
            risk += 12
        elif wind_speed > 10:
            risk += 6
        
        # Condition risk (0-25)
        condition_risks = {
            'Thunderstorm': 25,
            'Fog': 22,
            'Snow': 18,
            'Rain': 12,
            'Clouds': 5,
            'Clear': 0
        }
        risk += condition_risks.get(condition, 10)
        
        # Temperature risk (0-15)
        if temperature < -10 or temperature > 40:
            risk += 15
        elif temperature < 0 or temperature > 35:
            risk += 10
        elif temperature < 5:
            risk += 5
        
        return min(risk, 100)


def generate_weather_for_all_incidents():
    """Generate synthetic weather data for all incidents."""
    
    print("=" * 60)
    print("🌦️  AeroRisk - Synthetic Weather Generation")
    print("=" * 60)
    
    generator = WeatherGenerator(seed=42)
    
    # Get all incidents without weather data
    print("\n📊 Loading incidents...")
    
    query = text("""
        SELECT id, incident_date, severity, latitude, longitude
        FROM ingestion.incidents
        WHERE weather_data IS NULL
    """)
    
    with engine.connect() as conn:
        incidents = pd.read_sql(query, conn)
    
    print(f"   Found {len(incidents):,} incidents needing weather data")
    
    if len(incidents) == 0:
        print("   ✅ All incidents already have weather data!")
        return
    
    # Generate weather for each incident
    print(f"\n🌤️  Generating synthetic weather data...")
    
    weather_updates = []
    weather_records = []
    
    for idx, row in incidents.iterrows():
        try:
            weather = generator.generate_for_incident(
                severity=row['severity'],
                incident_date=row['incident_date'] if pd.notna(row['incident_date']) else datetime.now(),
                latitude=row['latitude'] if pd.notna(row['latitude']) else None,
                longitude=row['longitude'] if pd.notna(row['longitude']) else None
            )
            
            weather_updates.append({
                'id': str(row['id']),
                'weather_data': json.dumps(weather),
                'weather_conditions': weather['flight_conditions']
            })
            
            # Create weather_conditions record
            if pd.notna(row['incident_date']):
                weather_records.append({
                    'date': row['incident_date'].date(),
                    'hour': row['incident_date'].hour,
                    'airport_code': 'SYN',  # Synthetic
                    'temperature_c': weather['temperature_c'],
                    'feels_like_c': weather['feels_like_c'],
                    'humidity_percent': weather['humidity_percent'],
                    'pressure_hpa': weather['pressure_hpa'],
                    'visibility_m': weather['visibility_m'],
                    'cloud_cover_percent': weather['cloud_cover_percent'],
                    'wind_speed_kt': weather['wind_speed_kt'],
                    'wind_gust_kt': weather['wind_gust_kt'],
                    'wind_direction_deg': weather['wind_direction_deg'],
                    'weather_condition': weather['weather_condition'],
                    'severe_weather_flag': weather['severe_weather_flag'],
                    'weather_risk_score': weather['weather_risk_score'],
                    'source': 'SYNTHETIC'
                })
            
            if (idx + 1) % 5000 == 0:
                print(f"   Generated {idx + 1:,}/{len(incidents):,}...")
                
        except Exception as e:
            logger.error(f"Error for incident {row['id']}: {e}")
    
    print(f"   ✅ Generated weather for {len(weather_updates):,} incidents")
    
    # Batch update incidents
    print("\n💾 Updating incident records...")
    
    update_count = 0
    batch_size = 1000
    
    for i in range(0, len(weather_updates), batch_size):
        batch = weather_updates[i:i + batch_size]
        
        with engine.begin() as conn:
            for update in batch:
                conn.execute(text("""
                    UPDATE ingestion.incidents
                    SET weather_data = CAST(:weather_data AS jsonb),
                        weather_conditions = :weather_conditions,
                        updated_at = NOW()
                    WHERE id = CAST(:id AS uuid)
                """), update)
                update_count += 1
        
        if (i + batch_size) % 5000 == 0:
            print(f"   Updated {min(i + batch_size, len(weather_updates)):,}/{len(weather_updates):,}...")
    
    print(f"   ✅ Updated {update_count:,} incident records")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📊 Weather Generation Summary")
    print("=" * 60)
    
    # Get distribution
    with engine.connect() as conn:
        stats = pd.read_sql(text("""
            SELECT 
                weather_conditions,
                weather_data->>'weather_condition' as wx_type,
                COUNT(*) as count,
                AVG((weather_data->>'weather_risk_score')::float) as avg_risk
            FROM ingestion.incidents
            WHERE weather_data IS NOT NULL
            GROUP BY weather_conditions, weather_data->>'weather_condition'
            ORDER BY count DESC
        """), conn)
    
    print("\n📋 Weather Distribution:")
    print(stats.to_string(index=False))
    
    # Severity vs Weather correlation
    with engine.connect() as conn:
        correlation = pd.read_sql(text("""
            SELECT 
                severity,
                AVG((weather_data->>'weather_risk_score')::float) as avg_weather_risk,
                AVG((weather_data->>'visibility_m')::float) as avg_visibility,
                AVG((weather_data->>'wind_speed_kt')::float) as avg_wind
            FROM ingestion.incidents
            WHERE weather_data IS NOT NULL
            GROUP BY severity
            ORDER BY avg_weather_risk DESC
        """), conn)
    
    print("\n📈 Severity vs Weather Risk Correlation:")
    print(correlation.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✅ Synthetic Weather Generation Complete!")
    print("=" * 60)
    
    return len(weather_updates)


if __name__ == "__main__":
    generate_weather_for_all_incidents()
