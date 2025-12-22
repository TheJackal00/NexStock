import sqlite3
import random
from datetime import datetime, timedelta
import numpy as np

print("=" * 70)
print("REALISTIC TRANSACTION GENERATOR")
print("=" * 70)

# Connect to database
conn = sqlite3.connect('inventory.db')
cursor = conn.cursor()

# Get SKUs
cursor.execute('SELECT DISTINCT SKU FROM products ORDER BY SKU')
skus = [row[0] for row in cursor.fetchall()]

print(f"\nSKUs found: {skus}")

# Clear existing transactions
print("\n⚠️  Clearing existing transactions...")
cursor.execute('DELETE FROM TRANSACTIONS')
conn.commit()
print("✓ Transactions cleared")

# Define realistic daily demand patterns per SKU
# Based on typical retail electronics/product demand
demand_patterns = {
    '001': {'avg_daily': 15, 'std': 4},  # Popular item
    '002': {'avg_daily': 12, 'std': 3},  # Medium seller
    '003': {'avg_daily': 18, 'std': 5},  # High seller
    '004': {'avg_daily': 5, 'std': 2},   # Slow mover
    '005': {'avg_daily': 8, 'std': 3}    # Low-medium seller
}

# Generate data from 2024-01-01 to 2025-12-22 (today)
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 12, 22)
current_date = start_date

transactions = []
doc_counter = 100000

print(f"\nGenerating realistic transactions from {start_date.date()} to {end_date.date()}...")
print("Business rules:")
print("  • Daily sales: Normal distribution based on SKU patterns")
print("  • Weekly restocking: When inventory drops below threshold")
print("  • Supply = 110-120% of expected demand (realistic buffer)")

# Track inventory levels for reorder logic
inventory_levels = {sku: 500 for sku in skus}  # Start with 500 units each

day_count = 0
total_sales = 0
total_purchases = 0

while current_date <= end_date:
    day_count += 1
    
    # Generate sales for each SKU
    for sku in skus:
        pattern = demand_patterns[sku]
        
        # Generate daily demand (normal distribution, minimum 0)
        daily_demand = max(0, int(np.random.normal(pattern['avg_daily'], pattern['std'])))
        
        if daily_demand > 0:
            # Create sale transaction (negative volume)
            doc_number = f"INV-{doc_counter:06d}"
            doc_counter += 1
            
            transactions.append((
                current_date.strftime('%Y-%m-%d'),
                'Invoices',
                doc_number,
                sku,
                -daily_demand
            ))
            
            inventory_levels[sku] -= daily_demand
            total_sales += daily_demand
    
    # Weekly restock check (every Monday or when low)
    if current_date.weekday() == 0:  # Monday
        for sku in skus:
            pattern = demand_patterns[sku]
            # Reorder point: 2 weeks of average demand
            reorder_point = pattern['avg_daily'] * 14
            
            if inventory_levels[sku] < reorder_point:
                # Order quantity: 4 weeks of demand + 20% buffer
                order_qty = int(pattern['avg_daily'] * 28 * 1.2)
                
                # Create purchase transaction (positive volume)
                doc_number = f"PO-{doc_counter:06d}"
                doc_counter += 1
                
                transactions.append((
                    current_date.strftime('%Y-%m-%d'),
                    'Receipts',
                    doc_number,
                    sku,
                    order_qty
                ))
                
                inventory_levels[sku] += order_qty
                total_purchases += order_qty
    
    # Occasional returns/adjustments (1% of days)
    if random.random() < 0.01:
        sku = random.choice(skus)
        return_qty = random.randint(1, 5)
        
        doc_number = f"CN-{doc_counter:06d}"
        doc_counter += 1
        
        transactions.append((
            current_date.strftime('%Y-%m-%d'),
            'Credit Notes',
            doc_number,
            sku,
            return_qty  # Returns add back to inventory
        ))
        inventory_levels[sku] += return_qty
    
    current_date += timedelta(days=1)

print(f"\n✓ Generated {len(transactions)} transactions over {day_count} days")
print(f"  Total sales: {total_sales:,} units")
print(f"  Total purchases: {total_purchases:,} units")
print(f"  Net inventory: {total_purchases - total_sales:,} units ({((total_purchases/total_sales - 1) * 100):.1f}% buffer)")

# Insert into database
print("\nInserting into database...")
cursor.executemany('''
    INSERT INTO TRANSACTIONS (DATE, DOCUMENT_TYPE, DOC_NUMBER, SKU, VOLUME)
    VALUES (?, ?, ?, ?, ?)
''', transactions)

conn.commit()
print("✓ Transactions inserted")

# Verify and show summary
cursor.execute('SELECT MIN(DATE), MAX(DATE), COUNT(*) FROM TRANSACTIONS')
info = cursor.fetchone()
print(f"\n" + "=" * 70)
print("TRANSACTION SUMMARY")
print("=" * 70)
print(f"Date range: {info[0]} to {info[1]}")
print(f"Total transactions: {info[2]:,}")

print("\nBreakdown by SKU:")
print("-" * 70)
print(f"{'SKU':<6} {'Sales':<10} {'Purchases':<10} {'Net':<10} {'Buffer %':<10}")
print("-" * 70)

cursor.execute('''
    SELECT 
        SKU,
        SUM(CASE WHEN VOLUME < 0 THEN ABS(VOLUME) ELSE 0 END) as sales,
        SUM(CASE WHEN VOLUME > 0 THEN VOLUME ELSE 0 END) as purchases,
        SUM(VOLUME) as net
    FROM TRANSACTIONS
    GROUP BY SKU
    ORDER BY SKU
''')

for row in cursor.fetchall():
    sku, sales, purchases, net = row
    buffer_pct = ((purchases / sales - 1) * 100) if sales > 0 else 0
    print(f"{sku:<6} {int(sales):<10,} {int(purchases):<10,} {int(net):<10,} {buffer_pct:>9.1f}%")

print("-" * 70)

# Update inventory table to reflect final levels
print("\nUpdating inventory levels...")
for sku in skus:
    cursor.execute('UPDATE inventory SET VOLUME = ? WHERE SKU = ?', (inventory_levels[sku], sku))

conn.commit()
conn.close()

print("✓ Inventory levels updated")
print("\n" + "=" * 70)
print("✓ DONE! Realistic transactions generated successfully!")
print("=" * 70)
