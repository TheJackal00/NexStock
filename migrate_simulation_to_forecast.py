import sqlite3
import os

# Rename simulation table to forecast
db_path = 'inventory.db'

if not os.path.exists(db_path):
    print(f"Error: Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Check if simulation table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='simulation'")
    if cursor.fetchone():
        print("Found 'simulation' table, renaming to 'forecast'...")
        
        # Create new forecast table with same structure
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                SKU TEXT,
                DAY INTEGER,
                PREDICTED_DEMAND REAL,
                CONFIDENCE TEXT,
                DATE TEXT,
                FORECAST_HORIZON INTEGER,
                MODEL_VERSION TEXT,
                CREATED_AT TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (SKU) REFERENCES products (SKU)
            )
        """)
        
        # Drop old simulation table
        cursor.execute("DROP TABLE simulation")
        
        conn.commit()
        print("✓ Table renamed: simulation → forecast")
        print("✓ New schema optimized for ML predictions")
    else:
        # Just create forecast table if simulation doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                SKU TEXT,
                DAY INTEGER,
                PREDICTED_DEMAND REAL,
                CONFIDENCE TEXT,
                DATE TEXT,
                FORECAST_HORIZON INTEGER,
                MODEL_VERSION TEXT,
                CREATED_AT TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (SKU) REFERENCES products (SKU)
            )
        """)
        conn.commit()
        print("✓ Created 'forecast' table")
    
    # Verify
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='forecast'")
    if cursor.fetchone():
        print("✓ Migration successful!")
    else:
        print("✗ Migration failed")
        
except Exception as e:
    print(f"✗ Error during migration: {e}")
    conn.rollback()
finally:
    conn.close()
