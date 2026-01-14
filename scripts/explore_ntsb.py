"""
Explore NTSB MDB Database Structure
"""
import pyodbc

# Path to the MDB file
mdb_path = r"c:\Users\RaamGroup Digital\Downloads\United Airlines\aerorisk\data\raw\avall.mdb"

# Connection string for MS Access
conn_str = (
    r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
    f"DBQ={mdb_path};"
)

try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    print("=" * 60)
    print("📊 NTSB DATABASE STRUCTURE")
    print("=" * 60)
    
    # Get all tables
    tables = [row.table_name for row in cursor.tables(tableType='TABLE')]
    
    print(f"\n📁 Found {len(tables)} tables:\n")
    
    for table in tables:
        # Get row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
            count = cursor.fetchone()[0]
            print(f"   📋 {table}: {count:,} rows")
        except Exception as e:
            print(f"   📋 {table}: (error: {e})")
    
    # For main events table, show columns
    print("\n" + "=" * 60)
    print("📝 SAMPLE TABLE COLUMNS (events table if exists)")
    print("=" * 60)
    
    # Find the main events table
    main_tables = ['events', 'Events', 'EVENTS', 'aircraft', 'Aircraft']
    for table in main_tables:
        if table in tables:
            print(f"\n📋 Columns in '{table}':")
            for column in cursor.columns(table=table):
                print(f"   - {column.column_name} ({column.type_name})")
            break
    
    # Show first 3 rows of events
    if 'events' in [t.lower() for t in tables]:
        actual_table = [t for t in tables if t.lower() == 'events'][0]
        print(f"\n" + "=" * 60)
        print(f"📝 SAMPLE DATA FROM '{actual_table}' (first 3 rows)")
        print("=" * 60)
        cursor.execute(f"SELECT TOP 3 * FROM [{actual_table}]")
        columns = [desc[0] for desc in cursor.description]
        print(f"\nColumns: {columns[:10]}...")  # First 10 columns
        for row in cursor.fetchall():
            print(f"\nRow: {row[:5]}...")  # First 5 values
    
    conn.close()
    print("\n✅ Database exploration complete!")
    
except pyodbc.Error as e:
    print(f"❌ ODBC Error: {e}")
    print("\n💡 If you get a driver error, you may need to install:")
    print("   Microsoft Access Database Engine 2016 Redistributable")
    print("   https://www.microsoft.com/en-us/download/details.aspx?id=54920")
except Exception as e:
    print(f"❌ Error: {e}")
