"""
Create optimization_results table to store the latest optimization run results.
This allows /results page to display data without re-running optimization.
"""

import sqlite3

def create_optimization_table():
    conn = sqlite3.connect('inventory.db')
    c = conn.cursor()
    
    # Drop existing table if it exists
    c.execute('DROP TABLE IF EXISTS optimization_results')
    
    # Create optimization_results table
    c.execute('''
        CREATE TABLE optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT NOT NULL,
            sku TEXT NOT NULL,
            day INTEGER NOT NULL,
            order_quantity INTEGER DEFAULT 0,
            delivery_day INTEGER,
            total_cost REAL,
            total_orders INTEGER,
            service_level REAL,
            average_inventory REAL,
            stockout_days INTEGER,
            cost_breakdown TEXT,
            forecast_days INTEGER,
            shipping_cost REAL,
            holding_cost_rate REAL,
            stockout_penalty REAL
        )
    ''')
    
    # Create index for faster queries
    c.execute('CREATE INDEX idx_optimization_sku ON optimization_results(sku)')
    c.execute('CREATE INDEX idx_optimization_timestamp ON optimization_results(run_timestamp)')
    
    conn.commit()
    conn.close()
    
    print("✅ optimization_results table created successfully!")
    print("\nTable structure:")
    print("- Stores all optimization results per SKU")
    print("- Each row represents one day in the purchase schedule")
    print("- Truncated and repopulated on every /optimize run")
    print("- /results page reads from this table for display")

if __name__ == '__main__':
    create_optimization_table()
