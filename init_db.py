"""
Initialize database with sample data for Render deployment
"""
import sqlite3
from datetime import datetime, timedelta

def init_database():
    conn = sqlite3.connect('inventory.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS products
                 (SKU TEXT PRIMARY KEY, NAME TEXT, COST REAL, MARGIN REAL, EXPIRATION INTEGER)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS inventory
                 (SKU TEXT, VOLUME INTEGER, DATE TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (SKU TEXT, VOLUME INTEGER, DOCTYPE TEXT, DOCNUM TEXT, DATE TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS simulation
                 (SKU TEXT, Iteration INTEGER, Day INTEGER, DEMAND INTEGER, LEAD_TIME INTEGER, STOCK INTEGER)''')
    
    # Check if products already exist
    c.execute("SELECT COUNT(*) FROM products")
    if c.fetchone()[0] == 0:
        # Insert sample products
        products = [
            ('001', 'Product Alpha', 10.0, 0.30, 365),
            ('002', 'Product Beta', 15.0, 0.25, 180),
            ('003', 'Product Gamma', 8.0, 0.35, 90),
            ('004', 'Product Delta', 20.0, 0.20, 270),
            ('005', 'Product Epsilon', 12.0, 0.28, 365)
        ]
        c.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)
        
        # Insert sample inventory
        inventory = [
            ('001', 162, datetime.now().strftime('%Y-%m-%d')),
            ('002', 420, datetime.now().strftime('%Y-%m-%d')),
            ('003', 285, datetime.now().strftime('%Y-%m-%d')),
            ('004', 150, datetime.now().strftime('%Y-%m-%d')),
            ('005', 95, datetime.now().strftime('%Y-%m-%d'))
        ]
        c.executemany("INSERT INTO inventory VALUES (?,?,?)", inventory)
        
        # Insert sample transactions
        transactions = []
        for i in range(30):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            for sku in ['001', '002', '003', '004', '005']:
                # Incoming transaction
                transactions.append((sku, 50, 'Purchase Order', f'PO-{i:03d}', date))
                # Outgoing transaction
                transactions.append((sku, -30, 'Invoice', f'INV-{i:03d}', date))
        
        c.executemany("INSERT INTO transactions VALUES (?,?,?,?,?)", transactions)
        
        conn.commit()
        print("✅ Database initialized with sample data")
    else:
        print("ℹ️ Database already has data")
    
    conn.close()

if __name__ == '__main__':
    init_database()
