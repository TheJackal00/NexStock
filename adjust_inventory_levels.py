"""
Adjust inventory levels to realistic amounts that require replenishment
within the 90-day planning horizon.
Target: 15-25 days of stock coverage for demonstration purposes.
"""

import sqlite3
from datetime import datetime, timedelta

def adjust_inventory():
    conn = sqlite3.connect('inventory.db')
    c = conn.cursor()
    
    # Get average daily demand from forecast
    c.execute('''
        SELECT SKU, AVG(PREDICTED_DEMAND) as avg_daily_demand
        FROM forecast
        WHERE DAY <= 90
        GROUP BY SKU
    ''')
    
    demand_data = {row[0]: row[1] for row in c.fetchall()}
    
    print("=== ADJUSTING INVENTORY LEVELS ===\n")
    
    # Target coverage days for each SKU (varied for realism)
    target_coverage = {
        '001': 20,  # 20 days of stock
        '002': 15,  # 15 days of stock  
        '003': 25,  # 25 days of stock
        '004': 18,  # 18 days of stock
        '005': 22   # 22 days of stock
    }
    
    # Clear current inventory
    c.execute('DELETE FROM inventory')
    
    # Insert adjusted inventory levels
    for sku, avg_demand in demand_data.items():
        coverage_days = target_coverage.get(sku, 20)
        new_quantity = int(avg_demand * coverage_days)
        
        # Create a single batch entry
        batch_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get expiration days for this product
        c.execute('SELECT EXPIRATION FROM products WHERE SKU LIKE ?', (f'%{sku}%',))
        result = c.fetchone()
        expiration_days = result[0] if result else 365
        expiry_date = (datetime.now() + timedelta(days=expiration_days)).strftime('%Y-%m-%d')
        
        c.execute('''
            INSERT INTO inventory 
            (SKU, VOLUME, DATE, batch_id, expiry_date, location, reserved_qty)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (sku, new_quantity, batch_date, f'BATCH-{sku}-001', expiry_date, 'MAIN-WAREHOUSE', 0))
        
        print(f"SKU {sku}:")
        print(f"  - Avg Daily Demand: {avg_demand:.1f} units/day")
        print(f"  - Target Coverage: {coverage_days} days")
        print(f"  - New Inventory: {new_quantity:,} units")
        print(f"  - 90-day Demand: {avg_demand * 90:.0f} units")
        print(f"  - Needs to Order: {(avg_demand * 90 - new_quantity):.0f} units over 90 days")
        print()
    
    conn.commit()
    
    # Verify the changes
    print("\n=== VERIFICATION ===")
    c.execute('''
        SELECT i.SKU, SUM(i.VOLUME) as stock,
               (SELECT AVG(PREDICTED_DEMAND) FROM forecast WHERE SKU = i.SKU AND DAY <= 90) as avg_demand
        FROM inventory i
        GROUP BY i.SKU
        ORDER BY i.SKU
    ''')
    
    for row in c.fetchall():
        sku, stock, avg_demand = row
        coverage = stock / avg_demand if avg_demand > 0 else 0
        print(f"SKU {sku}: {stock:,} units ({coverage:.0f} days coverage)")
    
    conn.close()
    print("\n✅ Inventory levels adjusted successfully!")
    print("The optimizer should now generate purchase orders for all SKUs.")

if __name__ == '__main__':
    adjust_inventory()
