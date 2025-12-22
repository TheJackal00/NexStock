import requests
import sqlite3
from datetime import datetime
import json

# FRED API Configuration
# Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = "f263806d75cee9cb66204296d4f70980"  # Replace with your actual API key
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Economic indicators we'll track
INDICATORS = {
    'CPI': 'CPIAUCSL',           # Consumer Price Index (Inflation)
    'GDP': 'GDP',                 # Gross Domestic Product
    'INTEREST_RATE': 'FEDFUNDS', # Federal Funds Rate
    'UNEMPLOYMENT': 'UNRATE',     # Unemployment Rate
    'CONSUMER_CONF': 'UMCSENT',  # Consumer Sentiment Index
    'RETAIL_ELECTRONICS': 'RSEAS',    # Retail Sales: Electronics & Appliance Stores
    'RETAIL_ECOMMERCE': 'ECOMSA',     # E-commerce Retail Sales
    'RETAIL_TOTAL': 'RSXFS'           # Total Retail Sales (ex Food Services)
}

def create_economic_data_table():
    """Create table to store economic indicators"""
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ECONOMIC_DATA (
            DATE TEXT NOT NULL,
            INDICATOR TEXT NOT NULL,
            VALUE REAL NOT NULL,
            CREATED_AT TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (DATE, INDICATOR)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Economic data table created")

def fetch_fred_data(series_id, start_date='2024-12-01'):
    """Fetch data from FRED API"""
    
    if FRED_API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  ERROR: Please set your FRED API key")
        print("   Get it free at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start_date,
        'sort_order': 'asc'
    }
    
    try:
        response = requests.get(FRED_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'observations' in data:
            return data['observations']
        else:
            print(f"⚠️  No data for {series_id}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching {series_id}: {e}")
        return None

def store_economic_data(indicator_name, observations):
    """Store economic data in database"""
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    stored_count = 0
    for obs in observations:
        try:
            # Skip if value is '.'
            if obs['value'] == '.':
                continue
                
            cursor.execute('''
                INSERT OR REPLACE INTO ECONOMIC_DATA (DATE, INDICATOR, VALUE)
                VALUES (?, ?, ?)
            ''', (obs['date'], indicator_name, float(obs['value'])))
            stored_count += 1
        except (ValueError, KeyError) as e:
            continue
    
    conn.commit()
    conn.close()
    return stored_count

def insert_sample_economic_data():
    """Insert sample economic data for testing (monthly data for Dec 2024 - Sep 2025)"""
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    # Sample economic data (monthly values)
    sample_data = [
        # CPI (Consumer Price Index) - trending upward
        ('2024-12-01', 'CPI', 310.5), ('2025-01-01', 'CPI', 311.2), ('2025-02-01', 'CPI', 311.8),
        ('2025-03-01', 'CPI', 312.5), ('2025-04-01', 'CPI', 313.1), ('2025-05-01', 'CPI', 313.7),
        ('2025-06-01', 'CPI', 314.3), ('2025-07-01', 'CPI', 314.9), ('2025-08-01', 'CPI', 315.5),
        ('2025-09-01', 'CPI', 316.1),
        
        # GDP (Gross Domestic Product) - quarterly, growing
        ('2024-12-01', 'GDP', 27500), ('2025-03-01', 'GDP', 27800), 
        ('2025-06-01', 'GDP', 28100), ('2025-09-01', 'GDP', 28400),
        
        # Interest Rate (Federal Funds Rate) - decreasing
        ('2024-12-01', 'INTEREST_RATE', 5.25), ('2025-01-01', 'INTEREST_RATE', 5.00),
        ('2025-02-01', 'INTEREST_RATE', 4.75), ('2025-03-01', 'INTEREST_RATE', 4.75),
        ('2025-04-01', 'INTEREST_RATE', 4.50), ('2025-05-01', 'INTEREST_RATE', 4.50),
        ('2025-06-01', 'INTEREST_RATE', 4.25), ('2025-07-01', 'INTEREST_RATE', 4.25),
        ('2025-08-01', 'INTEREST_RATE', 4.00), ('2025-09-01', 'INTEREST_RATE', 4.00),
        
        # Unemployment Rate - stable
        ('2024-12-01', 'UNEMPLOYMENT', 4.2), ('2025-01-01', 'UNEMPLOYMENT', 4.1),
        ('2025-02-01', 'UNEMPLOYMENT', 4.0), ('2025-03-01', 'UNEMPLOYMENT', 3.9),
        ('2025-04-01', 'UNEMPLOYMENT', 3.9), ('2025-05-01', 'UNEMPLOYMENT', 3.8),
        ('2025-06-01', 'UNEMPLOYMENT', 3.8), ('2025-07-01', 'UNEMPLOYMENT', 3.9),
        ('2025-08-01', 'UNEMPLOYMENT', 4.0), ('2025-09-01', 'UNEMPLOYMENT', 4.0),
        
        # Consumer Confidence - improving
        ('2024-12-01', 'CONSUMER_CONF', 68.5), ('2025-01-01', 'CONSUMER_CONF', 70.2),
        ('2025-02-01', 'CONSUMER_CONF', 72.1), ('2025-03-01', 'CONSUMER_CONF', 73.5),
        ('2025-04-01', 'CONSUMER_CONF', 74.8), ('2025-05-01', 'CONSUMER_CONF', 76.2),
        ('2025-06-01', 'CONSUMER_CONF', 77.5), ('2025-07-01', 'CONSUMER_CONF', 78.1),
        ('2025-08-01', 'CONSUMER_CONF', 77.8), ('2025-09-01', 'CONSUMER_CONF', 77.2),
        
        # Retail Electronics Sales - seasonal pattern
        ('2024-12-01', 'RETAIL_ELECTRONICS', 8500), ('2025-01-01', 'RETAIL_ELECTRONICS', 7200),
        ('2025-02-01', 'RETAIL_ELECTRONICS', 6800), ('2025-03-01', 'RETAIL_ELECTRONICS', 7000),
        ('2025-04-01', 'RETAIL_ELECTRONICS', 7400), ('2025-05-01', 'RETAIL_ELECTRONICS', 7600),
        ('2025-06-01', 'RETAIL_ELECTRONICS', 7800), ('2025-07-01', 'RETAIL_ELECTRONICS', 8000),
        ('2025-08-01', 'RETAIL_ELECTRONICS', 8200), ('2025-09-01', 'RETAIL_ELECTRONICS', 7900),
        
        # E-commerce Retail Sales - growing
        ('2024-12-01', 'RETAIL_ECOMMERCE', 115000), ('2025-01-01', 'RETAIL_ECOMMERCE', 110000),
        ('2025-02-01', 'RETAIL_ECOMMERCE', 112000), ('2025-03-01', 'RETAIL_ECOMMERCE', 114000),
        ('2025-04-01', 'RETAIL_ECOMMERCE', 116000), ('2025-05-01', 'RETAIL_ECOMMERCE', 118000),
        ('2025-06-01', 'RETAIL_ECOMMERCE', 120000), ('2025-07-01', 'RETAIL_ECOMMERCE', 122000),
        ('2025-08-01', 'RETAIL_ECOMMERCE', 124000), ('2025-09-01', 'RETAIL_ECOMMERCE', 123000),
        
        # Total Retail Sales - steady growth
        ('2024-12-01', 'RETAIL_TOTAL', 685000), ('2025-01-01', 'RETAIL_TOTAL', 670000),
        ('2025-02-01', 'RETAIL_TOTAL', 675000), ('2025-03-01', 'RETAIL_TOTAL', 680000),
        ('2025-04-01', 'RETAIL_TOTAL', 685000), ('2025-05-01', 'RETAIL_TOTAL', 690000),
        ('2025-06-01', 'RETAIL_TOTAL', 695000), ('2025-07-01', 'RETAIL_TOTAL', 700000),
        ('2025-08-01', 'RETAIL_TOTAL', 705000), ('2025-09-01', 'RETAIL_TOTAL', 702000),
    ]
    
    for date, indicator, value in sample_data:
        cursor.execute('''
            INSERT OR REPLACE INTO ECONOMIC_DATA (DATE, INDICATOR, VALUE)
            VALUES (?, ?, ?)
        ''', (date, indicator, value))
    
    conn.commit()
    conn.close()
    return len(sample_data)

def fetch_all_indicators():
    """Fetch all economic indicators"""
    print("\n📊 Fetching Economic Indicators...\n")
    
    create_economic_data_table()
    
    # Check if API key is set
    if FRED_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️  No FRED API key configured")
        print("   Using sample economic data instead...\n")
        count = insert_sample_economic_data()
        print(f"✓ Inserted {count} sample data points")
    else:
        # Use real FRED API
        for name, series_id in INDICATORS.items():
            print(f"Fetching {name} ({series_id})...", end=' ')
            observations = fetch_fred_data(series_id)
            
            if observations:
                count = store_economic_data(name, observations)
                print(f"✓ Stored {count} data points")
            else:
                print("❌ Failed")
    
    # Show summary
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT INDICATOR, COUNT(*), MIN(DATE), MAX(DATE), 
               ROUND(AVG(VALUE), 2) as avg_value
        FROM ECONOMIC_DATA
        GROUP BY INDICATOR
        ORDER BY INDICATOR
    ''')
    
    print("\n" + "="*70)
    print("ECONOMIC DATA SUMMARY")
    print("="*70)
    for row in cursor.fetchall():
        print(f"{row[0]:20} {row[1]:4} points  |  {row[2]} to {row[3]}  |  Avg: {row[4]}")
    
    conn.close()
    print("\n✓ All economic data fetched and stored!")

if __name__ == "__main__":
    print("="*70)
    print("FRED Economic Data Fetcher")
    print("="*70)
    print("\nIndicators to fetch:")
    for name, series_id in INDICATORS.items():
        print(f"  • {name}: {series_id}")
    
    fetch_all_indicators()
