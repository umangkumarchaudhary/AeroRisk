"""
Test NTSB Fetcher - Quick verification
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.ntsb_fetcher import NTSBFetcher

# Initialize fetcher
mdb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "data", "raw", "avall.mdb")

print("=" * 60)
print("🛫 NTSB Fetcher Test")
print("=" * 60)
print(f"📁 MDB Path: {mdb_path}")

fetcher = NTSBFetcher(mdb_path)

# Get counts
event_count = fetcher.get_table_count('events')
aircraft_count = fetcher.get_table_count('aircraft')

print(f"📊 Events in database: {event_count:,}")
print(f"✈️  Aircraft records: {aircraft_count:,}")

# Fetch sample
print("\n📋 Fetching 5 sample events...")
df = fetcher.fetch_events(limit=5)
print(f"   Columns: {list(df.columns)[:10]}...")
print(f"   Sample ev_id: {df['ev_id'].tolist()}")

# Transform one
print("\n🔄 Testing transformation...")
sample_row = df.iloc[0]
transformed = fetcher.transform_event(sample_row)
print(f"   Transformed keys: {list(transformed.keys())}")
print(f"   Severity: {transformed['severity']}")
print(f"   Location: {transformed['location']}")
print(f"   Date: {transformed['incident_date']}")

print("\n✅ NTSB Fetcher test passed!")
