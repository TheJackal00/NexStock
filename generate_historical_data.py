import sqlite3
import random
from datetime import datetime, timedelta

# Connect to database
conn = sqlite3.connect('inventory.db')
cursor = conn.cursor()

# Get existing SKUs
cursor.execute('SELECT DISTINCT SKU FROM TRANSACTIONS ORDER BY SKU')
skus = [row[0] for row in cursor.fetchall()]

# Get current date range
cursor.execute('SELECT MIN(DATE), MAX(DATE), COUNT(*) FROM TRANSACTIONS')
date_info = cursor.fetchone()

print(f"Existing SKUs: {skus}")
print(f"Current date range: {date_info[0]} to {date_info[1]}")
print(f"Current transactions: {date_info[2]}")

# Document types with their characteristics
doc_types = {
    'Invoices': {'volume_range': (-50, -1), 'frequency': 0.4},  # Sales
    'Receipts': {'volume_range': (50, 300), 'frequency': 0.3},  # Purchases
    'Purchase Orders (PO)': {'volume_range': (0, 0), 'frequency': 0.1},  # No inventory impact
    'Credit Notes': {'volume_range': (-30, -1), 'frequency': 0.1},  # Returns
    'Debit Notes': {'volume_range': (-20, -1), 'frequency': 0.1}   # Adjustments
}

# Generate data from 2024-12-01 to 2025-09-29
start_date = datetime(2024, 12, 1)
end_date = datetime(2025, 9, 29)
current_date = start_date

transactions = []
doc_counter = 1000

print(f"\nGenerating transactions from {start_date.date()} to {end_date.date()}...")

while current_date <= end_date:
    # Generate 3-8 transactions per day
    num_transactions = random.randint(3, 8)
    
    for _ in range(num_transactions):
        sku = random.choice(skus)
        
        # Select document type based on frequency
        rand = random.random()
        cumulative = 0
        selected_doc_type = 'Invoices'
        
        for doc_type, props in doc_types.items():
            cumulative += props['frequency']
            if rand <= cumulative:
                selected_doc_type = doc_type
                break
        
        # Generate volume
        vol_range = doc_types[selected_doc_type]['volume_range']
        if vol_range[0] == 0 and vol_range[1] == 0:
            volume = 0
        else:
            volume = random.randint(vol_range[0], vol_range[1])
        
        # Generate document number
        doc_number = f"DOC-{doc_counter:06d}"
        doc_counter += 1
        
        transaction = (
            current_date.strftime('%Y-%m-%d'),
            selected_doc_type,
            doc_number,
            sku,
            volume
        )
        transactions.append(transaction)
    
    current_date += timedelta(days=1)

print(f"Generated {len(transactions)} transactions")

# Insert into database
print("Inserting into database...")
cursor.executemany('''
    INSERT INTO TRANSACTIONS (DATE, DOCUMENT_TYPE, DOC_NUMBER, SKU, VOLUME)
    VALUES (?, ?, ?, ?, ?)
''', transactions)

conn.commit()

# Verify
cursor.execute('SELECT MIN(DATE), MAX(DATE), COUNT(*) FROM TRANSACTIONS')
new_info = cursor.fetchone()
print(f"\nNew date range: {new_info[0]} to {new_info[1]}")
print(f"Total transactions: {new_info[2]}")

# Show breakdown by SKU
cursor.execute('''
    SELECT SKU, COUNT(*), SUM(VOLUME)
    FROM TRANSACTIONS
    WHERE DATE >= '2024-12-01'
    GROUP BY SKU
    ORDER BY SKU
''')
print("\nTransactions by SKU (new data only):")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} transactions, net volume: {row[2]}")

conn.close()
print("\nDone!")
