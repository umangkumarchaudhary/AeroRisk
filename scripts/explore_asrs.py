"""
Explore ASRS Excel Data - Using xlrd for .xls format
"""
import pandas as pd
import xlrd
import os

xls_path = r"c:\Users\RaamGroup Digital\Downloads\United Airlines\aerorisk\data\raw\ASRS_DBOnline.xls"

print("=" * 60)
print("📊 ASRS Database Exploration")
print("=" * 60)
print(f"📁 File: {xls_path}")
print(f"📏 Size: {os.path.getsize(xls_path) / (1024*1024):.2f} MB")

# Read with xlrd
print("\n⏳ Reading Excel file with xlrd...")
try:
    workbook = xlrd.open_workbook(xls_path)
    print(f"✅ Opened workbook with {workbook.nsheets} sheet(s)")
    
    for i, sheet_name in enumerate(workbook.sheet_names()):
        sheet = workbook.sheet_by_name(sheet_name)
        print(f"\n📋 Sheet {i+1}: '{sheet_name}'")
        print(f"   Rows: {sheet.nrows:,}, Columns: {sheet.ncols}")
        
        if sheet.nrows > 0:
            print(f"\n   Column Headers:")
            for col in range(min(sheet.ncols, 20)):  # First 20 columns
                try:
                    val = sheet.cell_value(0, col)
                    print(f"   {col+1}. {val}")
                except:
                    print(f"   {col+1}. (empty)")
            
            if sheet.ncols > 20:
                print(f"   ... and {sheet.ncols - 20} more columns")
            
            print(f"\n   Sample Row 2:")
            for col in range(min(sheet.ncols, 10)):
                try:
                    val = sheet.cell_value(1, col)
                    print(f"   {sheet.cell_value(0, col)}: {val}")
                except:
                    pass
                    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying pandas with different engines...")
    
    try:
        df = pd.read_excel(xls_path)
        print(f"✅ Loaded {len(df):,} rows")
        print(f"Columns: {list(df.columns)}")
    except Exception as e2:
        print(f"❌ Pandas error: {e2}")
