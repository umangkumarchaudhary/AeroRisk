"""
AeroRisk - Weather Data Fetcher
Fetch historical weather data from OpenWeather API to enrich incident data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from loguru import logger
from typing import Optional, Dict, List, Tuple
import json

from src.database.connection import engine
from sqlalchemy import text


class OpenWeatherClient:
    """Client for OpenWeather API."""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    HISTORY_URL = "https://history.openweathermap.org/data/2.5/history/city"
    ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.requests_made = 0
        self.last_request_time = None
        
    def _rate_limit(self):
        """Implement rate limiting (60 requests/minute for free tier)."""
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < 1.1:  # ~55 requests/minute max
                time.sleep(1.1 - elapsed)
        self.last_request_time = datetime.now()
        self.requests_made += 1
    
    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current weather for coordinates."""
        self._rate_limit()
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/weather",
                params={
                    'lat': lat,
                    'lon': lon,
                    'appid': self.api_key,
                    'units': 'metric'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Weather API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Weather fetch error: {e}")
            return None
    
    def get_historical_weather(self, lat: float, lon: float, dt: datetime) -> Optional[Dict]:
        """
        Get historical weather for coordinates at a specific time.
        Note: Historical API requires paid subscription for dates > 5 days ago.
        For free tier, we'll use current weather as a proxy for demo purposes.
        """
        self._rate_limit()
        
        # Convert datetime to Unix timestamp
        timestamp = int(dt.timestamp())
        
        try:
            # Try the One Call API 3.0 (requires subscription for history)
            response = requests.get(
                self.ONECALL_URL,
                params={
                    'lat': lat,
                    'lon': lon,
                    'dt': timestamp,
                    'appid': self.api_key,
                    'units': 'metric'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                # API key doesn't have historical access - use current as fallback
                return self.get_current_weather(lat, lon)
            else:
                logger.debug(f"Historical API returned {response.status_code}, using current weather")
                return self.get_current_weather(lat, lon)
                
        except Exception as e:
            logger.error(f"Historical weather fetch error: {e}")
            return None
    
    def parse_weather_data(self, data: Dict) -> Dict:
        """Parse OpenWeather response into our schema format."""
        if not data:
            return {}
        
        # Handle both current and historical response formats
        main = data.get('main', data.get('current', {}))
        wind = data.get('wind', {})
        clouds = data.get('clouds', {})
        weather = data.get('weather', [{}])[0] if 'weather' in data else {}
        
        # For historical one-call response
        if 'data' in data and isinstance(data['data'], list):
            hour_data = data['data'][0] if data['data'] else {}
            main = hour_data
            wind = hour_data
            clouds = hour_data
            weather = hour_data.get('weather', [{}])[0] if 'weather' in hour_data else {}
        
        # Calculate visibility in meters
        visibility = data.get('visibility', main.get('visibility', 10000))
        
        # Weather risk score calculation
        weather_risk = self._calculate_weather_risk(main, wind, visibility, weather)
        
        return {
            'temperature_c': main.get('temp'),
            'feels_like_c': main.get('feels_like'),
            'humidity_percent': main.get('humidity'),
            'pressure_hpa': main.get('pressure'),
            'visibility_m': visibility,
            'cloud_cover_percent': clouds.get('all') or main.get('clouds'),
            'wind_speed_kt': self._mps_to_knots(wind.get('speed', 0)),
            'wind_gust_kt': self._mps_to_knots(wind.get('gust', 0)) if wind.get('gust') else None,
            'wind_direction_deg': wind.get('deg'),
            'weather_condition': weather.get('main', 'Unknown'),
            'weather_description': weather.get('description', ''),
            'severe_weather_flag': self._is_severe_weather(weather, wind, visibility),
            'weather_risk_score': weather_risk,
            'raw_data': data
        }
    
    def _mps_to_knots(self, mps: float) -> float:
        """Convert meters per second to knots."""
        return mps * 1.94384 if mps else 0
    
    def _is_severe_weather(self, weather: Dict, wind: Dict, visibility: float) -> bool:
        """Determine if weather conditions are severe."""
        severe_conditions = ['Thunderstorm', 'Tornado', 'Squall', 'Hurricane']
        
        # Check weather type
        if weather.get('main') in severe_conditions:
            return True
        
        # High winds (> 25 knots)
        if wind.get('speed', 0) > 12.9:  # 12.9 m/s = 25 knots
            return True
        
        # Low visibility (< 1000m)
        if visibility and visibility < 1000:
            return True
        
        return False
    
    def _calculate_weather_risk(
        self, main: Dict, wind: Dict, visibility: float, weather: Dict
    ) -> float:
        """
        Calculate weather-based risk score (0-100).
        Higher = more dangerous conditions.
        """
        risk = 0.0
        
        # Visibility risk (0-30 points)
        if visibility:
            if visibility < 500:
                risk += 30
            elif visibility < 1000:
                risk += 25
            elif visibility < 3000:
                risk += 15
            elif visibility < 5000:
                risk += 10
            elif visibility < 8000:
                risk += 5
        
        # Wind risk (0-30 points)
        wind_speed = wind.get('speed', 0)
        wind_gust = wind.get('gust', 0)
        max_wind = max(wind_speed, wind_gust)
        
        if max_wind > 20:  # > 39 knots
            risk += 30
        elif max_wind > 15:  # > 29 knots
            risk += 25
        elif max_wind > 10:  # > 19 knots
            risk += 15
        elif max_wind > 7:  # > 14 knots
            risk += 10
        elif max_wind > 5:  # > 10 knots
            risk += 5
        
        # Weather condition risk (0-25 points)
        condition_risks = {
            'Thunderstorm': 25,
            'Tornado': 25,
            'Squall': 25,
            'Snow': 20,
            'Rain': 10,
            'Drizzle': 5,
            'Fog': 20,
            'Mist': 10,
            'Haze': 5,
            'Clouds': 3,
            'Clear': 0
        }
        condition = weather.get('main', 'Clear')
        risk += condition_risks.get(condition, 5)
        
        # Temperature extremes (0-15 points)
        temp = main.get('temp', 20)
        if temp:
            if temp < -10 or temp > 40:
                risk += 15
            elif temp < 0 or temp > 35:
                risk += 10
            elif temp < 5 or temp > 30:
                risk += 5
        
        return min(risk, 100)  # Cap at 100


def enrich_incidents_with_weather(api_key: str, batch_size: int = 100, max_incidents: int = 1000):
    """
    Fetch weather data for incidents that have coordinates.
    
    Args:
        api_key: OpenWeather API key
        batch_size: Number of incidents to process per batch
        max_incidents: Maximum total incidents to process (API rate limit consideration)
    """
    
    print("=" * 60)
    print("🌦️  AeroRisk - Weather Data Enrichment")
    print("=" * 60)
    
    client = OpenWeatherClient(api_key)
    
    # Get incidents with coordinates but no weather data
    print("\n📊 Finding incidents to enrich...")
    
    query = text("""
        SELECT id, incident_date, latitude, longitude, location, airport_code
        FROM ingestion.incidents
        WHERE latitude IS NOT NULL 
          AND longitude IS NOT NULL
          AND weather_data IS NULL
        ORDER BY incident_date DESC
        LIMIT :limit
    """)
    
    with engine.connect() as conn:
        incidents = pd.read_sql(query, conn, params={'limit': max_incidents})
    
    print(f"   Found {len(incidents):,} incidents needing weather data")
    
    if len(incidents) == 0:
        print("   ✅ All incidents already have weather data!")
        return
    
    # Process incidents
    print(f"\n🌤️  Fetching weather data (max {max_incidents} incidents)...")
    print(f"   Rate limit: ~55 requests/minute")
    
    enriched = 0
    failed = 0
    weather_records = []
    
    for idx, row in incidents.iterrows():
        try:
            # Fetch weather
            weather_data = client.get_historical_weather(
                row['latitude'], 
                row['longitude'],
                row['incident_date'] if pd.notna(row['incident_date']) else datetime.now()
            )
            
            if weather_data:
                parsed = client.parse_weather_data(weather_data)
                
                # Update incident with weather data
                update_query = text("""
                    UPDATE ingestion.incidents
                    SET weather_data = :weather_data,
                        weather_conditions = :weather_condition,
                        updated_at = NOW()
                    WHERE id = :incident_id
                """)
                
                with engine.begin() as conn:
                    conn.execute(update_query, {
                        'incident_id': str(row['id']),
                        'weather_data': json.dumps(parsed),
                        'weather_condition': 'IMC' if parsed.get('visibility_m', 10000) < 5000 else 'VMC'
                    })
                
                # Also store in weather_conditions table
                weather_record = {
                    'date': row['incident_date'].date() if pd.notna(row['incident_date']) else datetime.now().date(),
                    'hour': row['incident_date'].hour if pd.notna(row['incident_date']) else 12,
                    'airport_code': row['airport_code'] or 'UNK',
                    'temperature_c': parsed.get('temperature_c'),
                    'feels_like_c': parsed.get('feels_like_c'),
                    'humidity_percent': parsed.get('humidity_percent'),
                    'pressure_hpa': parsed.get('pressure_hpa'),
                    'visibility_m': parsed.get('visibility_m'),
                    'cloud_cover_percent': parsed.get('cloud_cover_percent'),
                    'wind_speed_kt': parsed.get('wind_speed_kt'),
                    'wind_gust_kt': parsed.get('wind_gust_kt'),
                    'wind_direction_deg': parsed.get('wind_direction_deg'),
                    'weather_condition': parsed.get('weather_condition'),
                    'severe_weather_flag': parsed.get('severe_weather_flag', False),
                    'weather_risk_score': parsed.get('weather_risk_score'),
                    'source': 'OpenWeather',
                    'raw_data': json.dumps(parsed.get('raw_data', {}))
                }
                weather_records.append(weather_record)
                
                enriched += 1
            else:
                failed += 1
            
            # Progress update
            if (enriched + failed) % 10 == 0:
                print(f"   Progress: {enriched + failed}/{len(incidents)} ({enriched} enriched, {failed} failed)")
                print(f"   API requests made: {client.requests_made}")
            
        except Exception as e:
            logger.error(f"Error enriching incident {row['id']}: {e}")
            failed += 1
    
    # Bulk insert weather records
    if weather_records:
        print(f"\n💾 Saving {len(weather_records)} weather records to database...")
        weather_df = pd.DataFrame(weather_records)
        
        try:
            weather_df.to_sql(
                'weather_conditions',
                engine,
                schema='ingestion',
                if_exists='append',
                index=False,
                method='multi'
            )
            print(f"   ✅ Saved to ingestion.weather_conditions")
        except Exception as e:
            print(f"   ⚠️  Could not save to weather_conditions table: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Weather Enrichment Summary")
    print("=" * 60)
    print(f"""
    Incidents processed: {enriched + failed}
    Successfully enriched: {enriched}
    Failed: {failed}
    API requests made: {client.requests_made}
    
    Weather conditions added to:
    - ingestion.incidents (weather_data, weather_conditions columns)
    - ingestion.weather_conditions table
    """)
    
    # Show sample of enriched data
    if enriched > 0:
        print("\n📋 Sample enriched incident:")
        sample_query = text("""
            SELECT id, location, incident_date, weather_conditions,
                   weather_data->>'weather_condition' as wx_type,
                   weather_data->>'weather_risk_score' as risk_score,
                   weather_data->>'visibility_m' as visibility
            FROM ingestion.incidents
            WHERE weather_data IS NOT NULL
            LIMIT 5
        """)
        with engine.connect() as conn:
            sample = pd.read_sql(sample_query, conn)
            print(sample.to_string())
    
    print("\n" + "=" * 60)
    print("✅ Weather Enrichment Complete!")
    print("=" * 60)
    
    return enriched


if __name__ == "__main__":
    # Get API key from environment or use provided one
    API_KEY = os.getenv("OPENWEATHER_API_KEY", "b55a3c3f6e3a08d127e6fc482500b8af")
    
    # Test the API first
    print("🔑 Testing OpenWeather API connection...")
    client = OpenWeatherClient(API_KEY)
    test_result = client.get_current_weather(40.7128, -74.0060)  # NYC
    
    if test_result:
        print(f"   ✅ API working! Current NYC weather: {test_result.get('weather', [{}])[0].get('description', 'N/A')}")
        print(f"   Temperature: {test_result.get('main', {}).get('temp', 'N/A')}°C")
        
        # Run enrichment
        enriched = enrich_incidents_with_weather(API_KEY, max_incidents=500)
    else:
        print("   ❌ API connection failed! Check your API key.")
