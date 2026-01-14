"""
Load NTSB Data - Small Test Batch
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.ntsb_fetcher import NTSBFetcher
from src.database.connection import engine
from sqlalchemy import text

# Path to MDB
mdb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "data", "raw", "avall.mdb")

print("=" * 60)
print("🧪 NTSB Data Load - TEST (100 records)")
print("=" * 60)

fetcher = NTSBFetcher(mdb_path)

# Load just 100 records first
print("Loading 100 test records...")
loaded = fetcher.load_to_database(limit=100, batch_size=50)

print(f"\n✅ Loaded {loaded} records")

# Verify
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM ingestion.incidents"))
    count = result.fetchone()[0]
    print(f"📊 Total in database: {count}")
    
    result = conn.execute(text("SELECT severity, COUNT(*) FROM ingestion.incidents GROUP BY severity"))
    print("\nSeverity distribution:")
    for row in result.fetchall():
        print(f"  {row[0]}: {row[1]}")
