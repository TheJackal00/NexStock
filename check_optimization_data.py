import sqlite3

conn = sqlite3.connect('inventory.db')
c = conn.cursor()

# First check the inventory table structure
c.execute('PRAGMA table_info(inventory)')
print("=== INVENTORY TABLE STRUCTURE ===")
for row in c.fetchall():
    print(row)

print("\n=== INVENTORY LEVELS ===")
c.execute('SELECT SKU, SUM(VOLUME) as total_stock FROM inventory GROUP BY SKU ORDER BY SKU')
inventory_data = {}
for row in c.fetchall():
    sku, qty = row
    inventory_data[sku] = qty
    print(f"SKU {sku}: {qty:,.0f} units in stock")

print("\n=== FORECAST SUMMARY (90 days) ===")
c.execute('''
    SELECT SKU, COUNT(*) as days, 
           AVG(PREDICTED_DEMAND) as avg_demand,
           SUM(PREDICTED_DEMAND) as total_demand_90d
    FROM forecast 
    WHERE DAY <= 90
    GROUP BY SKU 
    ORDER BY SKU
''')
for row in c.fetchall():
    sku, days, avg_demand, total_demand = row
    initial_stock = inventory_data.get(sku, 0)
    coverage_days = initial_stock / avg_demand if avg_demand > 0 else 999
    print(f"SKU {sku}:")
    print(f"  - Initial Stock: {initial_stock:,} units")
    print(f"  - 90-day Demand: {total_demand:.0f} units")
    print(f"  - Avg Daily Demand: {avg_demand:.1f} units/day")
    print(f"  - Stock Coverage: {coverage_days:.0f} days")
    print(f"  - Needs Replenishment: {'NO - Stock exceeds 90 days' if coverage_days > 90 else 'YES'}")
    print()

print("=== PRODUCT DETAILS ===")
c.execute('SELECT SKU, NAME, COST, MARGIN, EXPIRATION FROM products ORDER BY SKU')
for row in c.fetchall():
    sku, name, cost, margin, expiration = row
    print(f"SKU {sku} ({name}):")
    print(f"  - Unit Cost: ${cost:.2f}")
    print(f"  - Margin: {margin*100:.0f}%")
    print(f"  - Expiration: {expiration} days")
    print()

conn.close()
