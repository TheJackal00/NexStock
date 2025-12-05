import requests

# Test with restarted server
print("Testing APIs after server restart...\n")

# Test overview API
r = requests.get('http://localhost:5000/api/analytics/overview')
data = r.json()

turnover_skus = sorted(set([item['SKU'] for item in data['turnover_data']]))
print(f"✓ Turnover Chart - Unique SKUs: {turnover_skus}")
print(f"  Total turnover records: {len(data['turnover_data'])}")

# Test trends API
r2 = requests.get('http://localhost:5000/api/analytics/trends')
data2 = r2.json()

print(f"\n✓ Demand Analysis - SKUs: {sorted([item['SKU'] for item in data2['demand_analysis']])}")
print(f"\nTransaction counts per SKU:")
for item in sorted(data2['demand_analysis'], key=lambda x: x['SKU']):
    print(f"  SKU {item['SKU']}: {item['transaction_count']} transactions, {item['total_volume']} total volume")
