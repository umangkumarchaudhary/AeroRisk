"""
Detailed NTSB Database Schema Explorer
"""
import pyodbc

mdb_path = r"c:\Users\RaamGroup Digital\Downloads\United Airlines\aerorisk\data\raw\avall.mdb"

conn_str = (
    r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
    f"DBQ={mdb_path};"
)

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

print("=" * 80)
print("📊 NTSB DATABASE - COMPLETE SCHEMA")
print("=" * 80)

# Get all tables with row counts
tables = [row.table_name for row in cursor.tables(tableType='TABLE')]

table_info = []
for table in tables:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
        count = cursor.fetchone()[0]
        table_info.append((table, count))
    except:
        table_info.append((table, 0))

# Sort by row count
table_info.sort(key=lambda x: x[1], reverse=True)

print(f"\n📁 {len(tables)} Tables (sorted by size):\n")
for table, count in table_info:
    print(f"   {table:25} → {count:>10,} rows")

# Get columns for key tables
key_tables = ['events', 'aircraft', 'Flight_Crew', 'injury', 'Findings', 'Narratives']

print("\n" + "=" * 80)
print("📝 KEY TABLE SCHEMAS")
print("=" * 80)

for table_name in key_tables:
    # Find exact table name (case insensitive)
    actual = next((t for t in tables if t.lower() == table_name.lower()), None)
    if actual:
        print(f"\n📋 {actual}")
        print("-" * 40)
        for col in cursor.columns(table=actual):
            print(f"   {col.column_name:30} {col.type_name}")

# Sample from events table
print("\n" + "=" * 80)
print("📝 SAMPLE EVENTS DATA")
print("=" * 80)

cursor.execute("SELECT TOP 5 * FROM [events]")
columns = [desc[0] for desc in cursor.description]
print(f"\nColumns ({len(columns)} total):")
print(columns)

print("\n\nFirst 5 events:")
rows = cursor.fetchall()
for i, row in enumerate(rows):
    print(f"\n--- Event {i+1} ---")
    for j, (col, val) in enumerate(zip(columns, row)):
        if val is not None and str(val).strip():
            print(f"  {col}: {val}")

conn.close()
